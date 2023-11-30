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
from turtle import forward

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.auxilary import *

import torch.optim as optim
from sklearn.linear_model import Ridge

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.eval()

class exp_net(torch.nn.Module):
    def __init__(self, feature_dim, feature_len):
        super().__init__()
        self.fc1 = torch.nn.Linear(feature_dim, 1)
        self.fc2 = torch.nn.Linear(feature_len, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x.permute(0,2,1))
        return x


class Calculator :
  def __init__(self) :
    self.data = {}
    self.space = []
    self.cost = {}
    self.cost['cor'] = 0
    self.cost['sub'] = 1
    self.cost['del'] = 1
    self.cost['ins'] = 1
  def calculate(self, lab, rec) :
    # Initialization
    lab.insert(0, '')
    rec.insert(0, '')
    while len(self.space) < len(lab) :
      self.space.append([])
    for row in self.space :
      for element in row :
        element['dist'] = 0
        element['error'] = 'non'
      while len(row) < len(rec) :
        row.append({'dist' : 0, 'error' : 'non'})
    for i in range(len(lab)) :
      self.space[i][0]['dist'] = i
      self.space[i][0]['error'] = 'del'
    for j in range(len(rec)) :
      self.space[0][j]['dist'] = j
      self.space[0][j]['error'] = 'ins'
    self.space[0][0]['error'] = 'non'
    for token in lab :
      if token not in self.data and len(str(token)) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    for token in rec :
      if token not in self.data and len(str(token)) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    # Computing edit distance
    for i, lab_token in enumerate(lab) :
      for j, rec_token in enumerate(rec) :
        if i == 0 or j == 0 :
          continue
        min_dist = sys.maxsize
        min_error = 'none'
        dist = self.space[i-1][j]['dist'] + self.cost['del']
        error = 'del'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        dist = self.space[i][j-1]['dist'] + self.cost['ins']
        error = 'ins'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        if lab_token == rec_token :
          dist = self.space[i-1][j-1]['dist'] + self.cost['cor']
          error = 'cor'
        else :
          dist = self.space[i-1][j-1]['dist'] + self.cost['sub']
          error = 'sub'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        self.space[i][j]['dist'] = min_dist
        self.space[i][j]['error'] = min_error
    # Tracing back
    result = {'lab':[], 'rec':[], 'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    i = len(lab) - 1
    j = len(rec) - 1
    while True :
      if self.space[i][j]['error'] == 'cor' : # correct
        if len(str(lab[i])) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
          result['all'] = result['all'] + 1
          result['cor'] = result['cor'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'sub' : # substitution
        if len(str(lab[i])) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
          result['all'] = result['all'] + 1
          result['sub'] = result['sub'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'del' : # deletion
        if len(str(lab[i])) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
          result['all'] = result['all'] + 1
          result['del'] = result['del'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, "")
        i = i - 1
      elif self.space[i][j]['error'] == 'ins' : # insertion
        if len(str(rec[j])) > 0 :
          self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
          result['ins'] = result['ins'] + 1
        result['lab'].insert(0, "")
        result['rec'].insert(0, rec[j])
        j = j - 1
      elif self.space[i][j]['error'] == 'non' : # starting point
        break
      else : # shouldn't reach here
        print('this should not happen , i = {i} , j = {j} , error = {error}'.format(i = i, j = j, error = self.space[i][j]['error']))
    return result
  def overall(self) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in self.data :
      result['all'] = result['all'] + self.data[token]['all']
      result['cor'] = result['cor'] + self.data[token]['cor']
      result['sub'] = result['sub'] + self.data[token]['sub']
      result['ins'] = result['ins'] + self.data[token]['ins']
      result['del'] = result['del'] + self.data[token]['del']
    return result
  def cluster(self, data) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in data :
      if token in self.data :
        result['all'] = result['all'] + self.data[token]['all']
        result['cor'] = result['cor'] + self.data[token]['cor']
        result['sub'] = result['sub'] + self.data[token]['sub']
        result['ins'] = result['ins'] + self.data[token]['ins']
        result['del'] = result['del'] + self.data[token]['del']
    return result
  def keys(self) :
      return list(self.data.keys())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', default='/media/DATA/stl/wenet/examples/librispeech/mini/exp/sp_spec_aug_identity_full/train.yaml', help='config file')
    parser.add_argument('--test_data', default='/media/DATA/stl/wenet/examples/librispeech/full_model/data/test_clean/format.data', help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', default='/media/DATA/stl/wenet/examples/librispeech/mini/exp/sp_spec_aug_identity_full/avg_30.pt', help='checkpoint model')
    parser.add_argument('--dict', default='/media/DATA/stl/wenet/examples/librispeech/full_model/data/lang_char/train_960_unigram5000_units.txt', help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', default='/media/DATA/stl/wenet/examples/librispeech/full_model/exp/words_in_paper/', help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='ctc_greedy_search',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=1,
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
    parser.add_argument('--modality', default='AO', help='AO, VO or AV')
    parser.add_argument('--word', required=False, default=[1189, 2653, 522, 4442, 
        4958, 2100, 2101, 401, 1173, 385, 2021, 4532, 586, 3553, 3043, 3044, 4520, 
        964, 3758, 400, 4437, 2055, 2033, 4936, 1733, 1763])
    # parser.add_argument('--word', required=False, default=[3879, 3885, 1634, 1641, 2034])
    # parser.add_argument('--word', required=False, default=[3828, 4895, 607])
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
    model = init_asr_model(configs)

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    # print(model.state_dict()[0])
    # exit()
    # model.load_lm()
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = model.to(device)

    model.eval()
    # model.train()
    # model.apply(apply_dropout)
    calculator = Calculator()

    for batch_idx, batch in enumerate(test_data_loader):
        keys, feats, target, feats_lengths, target_lengths = batch
        if keys[0] not in ['1320-122617-0038']:
        # if keys[0] not in ['237-134500-0023']:
            continue
        feats = feats.to(device)#.requires_grad_(True)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        target_lengths = target_lengths.to(device)
        xs_l, pos_emb, masks = model.encoder.prepare(feats, feats_lengths, None)
        xs_feat_l = np.ones([xs_l.shape[1]])
        hyps_gt, _, _ = model.ctc_greedy_search_with_removal(
                xs_l, pos_emb, masks,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                simulate_streaming=args.simulate_streaming)
        exp_features = [np.ones(xs_l.shape[1])]
        exp_labels = [1]
        exp_dist = [1.]
        # for remove_i in range(xs_l.shape[1]):
        #     xs, pos_emb, masks = model.encoder.prepare(feats, feats_lengths, remove_i)
        #     xs_feat = np.ones([xs_l.shape[1]])
        #     xs_feat[remove_i] = 0
        # remove_i = 0
        # stp = 3
        # while remove_i <= xs_l.shape[1] - stp: 
        #     xs, pos_emb, masks = model.encoder.prepare(feats, feats_lengths, None)
        #     xs[:, remove_i: remove_i+stp, :] = 0
        #     remove_i += 1
        token_id = 22
        for remove_i in range(1000):
            xs, pos_emb, masks = model.encoder.prepare(feats, feats_lengths, None)
            xs_feat = np.ones([xs_l.shape[1]])
            for i in range(xs.shape[1]):
            #     xs[:, i, :] *= np.random.rand() * 2
                if (np.random.rand() < 0.5):# and (i != token_id):
                    xs[:, i, :] *= 0
                    xs_feat[i] *= 0
            hyps, ctc_probs, predIdx = model.ctc_greedy_search_with_removal(
                xs, pos_emb, masks,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                simulate_streaming=args.simulate_streaming)
            result = calculator.calculate(hyps_gt[0], hyps[0])
            wer = float(result['ins'] + result['sub'] + result['del']) * 1. / result['all']
            exp_features.append(xs_feat)
            # exp_labels.append(1 if 2033 in predIdx[0] else 0)
            # exp_labels.append(1 if predIdx[0][token_id]==2033 else 0)
            exp_labels.append(ctc_probs[0][token_id][2033].exp()-ctc_probs[0][token_id][2055].exp())
            # exp_labels.append(1 if predIdx[0][9]==4958 else 0)
            # exp_labels.append(wer)
            # exp_labels.append(1 if wer<0.5 else 0)
            cos_dist = (xs_feat * xs_feat_l).sum()/np.sqrt((xs_feat ** 2).sum())/np.sqrt((xs_feat_l ** 2).sum())
            exp_dist.append(np.exp((-cos_dist)))
            print(remove_i, wer)

        explanation = exp_net(xs_l.shape[2], xs_l.shape[1])
        explanation.to(device)
        explanation.train()
        optimizer = optim.Adam(explanation.parameters())
        # data = torch.cat(exp_features).to(device)
        data = torch.tensor(exp_features).to(device)
        # gt = torch.stack(exp_labels).to(device)
        gt = torch.tensor(exp_labels, dtype=torch.float64).to(device)
        dist = torch.tensor(exp_dist).to(device)

        # x = ((data.sum(2) > 0) * 1.0)
        x = data
        theta = torch.linalg.inv(x.T * dist @ x) @ x.T * dist @ gt
        # theta = torch.linalg.inv(x.T @ x) @ x.T @ gt
        data.sum(2)@theta - gt
        
        for train_iter in range(100):
            predictions = explanation(data)
            loss = ((predictions - gt) ** 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(train_iter, loss)
        # loss, loss_att, loss_ctc = model(feats, feats_lengths,
        #                                     target, target_lengths)
        
        # interpret_by_word(model, ctc_probs, predIdx, char_dict, data=(keys, feats, feats_lengths, target, target_lengths), tgtWord=args.word, base_folder=args.result_file)
        # print(hyps)
