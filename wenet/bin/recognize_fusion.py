# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint1', required=True, help='checkpoint model')
    parser.add_argument('--checkpoint2', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    # Init dataset and data loader
    test_collate_conf = copy.deepcopy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_conf['spec_sub'] = False
    test_collate_conf['feature_dither'] = False
    test_collate_conf['speed_perturb'] = False
    if raw_wav:
        test_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
        test_collate_conf['wav_distortion_conf']['wav_dither'] = 0.0
    test_collate_func = CollateFunc(**test_collate_conf, raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False
    test_dataset = AudioDataset(args.test_data,
                                **dataset_conf,
                                raw_wav=raw_wav)
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=0)

    # Init asr model from configs
    model1 = init_asr_model(configs)
    model2 = init_asr_model(configs)

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    load_checkpoint(model1, args.checkpoint1)
    load_checkpoint(model2, args.checkpoint2)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model1 = model1.to(device)
    model2 = model2.to(device)

    model1.eval()
    model2.eval()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            if args.mode == 'attention':
                hyps = model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'ctc_greedy_search':
                # model 1
                batch_size = feats.shape[0]
                encoder_out, encoder_mask = model1._forward_encoder(
                    feats, feats_lengths, args.decoding_chunk_size,
                    args.num_decoding_left_chunks,
                    args.simulate_streaming)  # (B, maxlen, encoder_dim)
                maxlen = encoder_out.size(1)
                encoder_out_lens = encoder_mask.squeeze(1).sum(1)
                ctc_probs1 = model1.ctc.log_softmax(
                    encoder_out)  # (B, maxlen, vocab_size)
                # model 2
                encoder_out, encoder_mask = model2._forward_encoder(
                    feats, feats_lengths, args.decoding_chunk_size,
                    args.num_decoding_left_chunks,
                    args.simulate_streaming)  # (B, maxlen, encoder_dim)
                maxlen = encoder_out.size(1)
                encoder_out_lens = encoder_mask.squeeze(1).sum(1)
                ctc_probs2 = model2.ctc.log_softmax(
                    encoder_out)  # (B, maxlen, vocab_size)

                # ctc_probs1 = ctc_probs1.exp()
                # ctc_probs2 = ctc_probs2.exp()
                ctc_probs_mat = torch.zeros(ctc_probs1.shape[1], ctc_probs1.shape[1])
                ctc_codes_mat = torch.zeros(ctc_probs1.shape[1], ctc_probs1.shape[1])
                for i in range(ctc_probs1.shape[1]):
                    for j in range(ctc_probs2.shape[1]):
                        ctc_probs_mat[i, j] = ((ctc_probs1[0, i, :] + ctc_probs2[0, j, :]) / 2).max()
                        ctc_codes_mat[i, j] = ((ctc_probs1[0, i, :] + ctc_probs2[0, j, :]) / 2).argmax()

                hyps = torch.ones(1, ctc_probs1.shape[1], 1).cuda() * (-1)
                i, j = 0, 0
                for k in range(ctc_probs1.shape[1]):
                    # hyp_mat = ctc_codes_mat[i: i + 2, j: j + 2]
                    prob_mat = ctc_probs_mat[i: i + 2, j: j + 2]
                    prob_mat[0, 0] = 0
                    hyps[0, k, 0] = ctc_codes_mat[i, j]
                    if prob_mat.argmax() == 1:
                        j += 1
                    elif prob_mat.argmax() == 2:
                        i += 1
                    else:
                        i += 1
                        j += 1
                topk_index = hyps

                # # ctc_probs = (ctc_probs1 + ctc_probs2) / 2
                # # ctc_probs = ((ctc_probs1.exp() + ctc_probs2.exp()) / 2).log()
                # # ctc_probs = torch.cat((ctc_probs1, ctc_probs2), 0).max(0, keepdims=True)[0]

                # topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
                topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
                mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
                topk_index = topk_index.masked_fill_(mask, model1.eos)  # (B, maxlen)
                hyps = [hyp.tolist() for hyp in topk_index]
                predIdx = [hyp.tolist() for hyp in topk_index]
                hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == 'ctc_prefix_beam_search':
                assert (feats.size(0) == 1)
                hyp = model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'attention_rescoring':
                assert (feats.size(0) == 1)
                hyp = model.attention_rescoring(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight)
                hyps = [hyp]
            for i, key in enumerate(keys):
                content = ''
                for w in hyps[i]:
                    if w == eos:
                        break
                    content += char_dict[w]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))
