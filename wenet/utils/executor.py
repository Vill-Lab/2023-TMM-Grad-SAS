# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
from contextlib import nullcontext
# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_
# from examples.librispeech.viz.wenet.auxilary import get_mask
from wenet.auxilary import *

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        num_seen_utts = 0
        num_total_batch = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            key, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)#.requires_grad_(True)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            num_utts = target_lengths.size(0)
            if num_utts == 0:
                continue
            context = None
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if is_distributed and batch_idx % accum_grad != 0:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext
            
            with context():
                with torch.cuda.amp.autocast(scaler is not None):
                    if args['train_method'] == 'diags':
                        ctc_probs = model.ctc_greedy_search(feats, feats_lengths, bp=True)
                        ctc_probs.backward(retain_graph=True)
                        dr = get_diags(model, diags=[-3,-2,-1,0,1,2,3])
                        optimizer.zero_grad()
                        loss = (args['diag_ratio'] - dr).clamp(0) * 100
                        # loss = -dr.log()
                        loss_ctc, loss_att = None, None
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    else:
                        loss, loss_att, loss_ctc = model(feats, feats_lengths,
                                                            target, target_lengths)
                        loss = loss / accum_grad
                        
                if args['train_method'] == 'normal':
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                elif args['train_method'] == 'drop_mask':
                # autocast context
                # The more details about amp can be found in
                # https://pytorch.org/docs/stable/notes/amp_examples.html
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    drop_masks_v = get_mask(model)
                    # drop_masks = drop_masks_v[drop_masks_v > drop_masks_v.max() * 0.8]
                    # v, drop_masks = torch.topk(drop_masks_v, int(len(drop_masks_v) * args['att_dropout_rate']))
                    v, drop_masks = torch.topk(drop_masks_v, int(len(drop_masks_v)))
                    drop_masks = drop_masks[v > v.max() * 0.9]
                    # optimizer.zero_grad()
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss, loss_att, loss_ctc = model(feats, feats_lengths,
                                                            target, target_lengths, drop_masks)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                elif args['train_method'] == 'drop_mask_nograd':
                # autocast context
                # The more details about amp can be found in
                # https://pytorch.org/docs/stable/notes/amp_examples.html
                    # if use_amp:
                    #     scaler.scale(loss).backward()
                    # else:
                    #     loss.backward()
                    drop_masks_v = get_mask_nograd(model)
                    # drop_masks = drop_masks_v[drop_masks_v > drop_masks_v.max() * 0.8]
                    # v, drop_masks = torch.topk(drop_masks_v, int(len(drop_masks_v) * args['att_dropout_rate']))
                    v, drop_masks = torch.topk(drop_masks_v, int(len(drop_masks_v)))
                    drop_masks = drop_masks[v > v.max() * 0.9]
                    # optimizer.zero_grad()
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss, loss_att, loss_ctc = model(feats, feats_lengths,
                                                            target, target_lengths, drop_masks)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                elif args['train_method'] == 'reg_mask':
                # autocast context
                # The more details about amp can be found in
                # https://pytorch.org/docs/stable/notes/amp_examples.html

                    # if use_amp:
                    #     scaler.scale(loss).backward()
                    # else:
                    #     loss.backward()
                    reg = get_mask_nograd(model)
                    # optimizer.zero_grad()
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    # with torch.cuda.amp.autocast(scaler is not None):
                    #     loss, loss_att, loss_ctc = model(feats, feats_lengths,
                    #                                         target, target_lengths)
                    #     loss = (loss + args['reg_lmbd'] * reg.max()) / accum_grad
                    if use_amp:
                        scaler.scale(loss + args['reg_lmbd'] * reg.max() / accum_grad).backward()
                    else:
                        (loss + args['reg_lmbd'] * reg.max() / accum_grad).backward()
                        
                elif args['train_method'] == 'ctc_reg':
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss, loss_att, loss_ctc = model(feats, feats_lengths,
                                                            target, target_lengths)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    drop_masks_v = get_mask_ctc_reg(model)
                    # drop_masks = drop_masks_v[drop_masks_v > drop_masks_v.max() * 0.8]
                    # v, drop_masks = torch.topk(drop_masks_v, int(len(drop_masks_v) * args['att_dropout_rate']))
                    v, drop_masks = torch.topk(drop_masks_v, int(0.1*len(drop_masks_v)))
                    drop_masks = drop_masks[v > v.max() * 0.9]
                    # optimizer.zero_grad()
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss, loss_att, loss_ctc, score_mat = model(feats, feats_lengths,
                                                            target, target_lengths, drop_masks, score=True)
                        regloss = (score_mat[..., 0]-0.7).clamp(0).sum()/score_mat.shape[0]/score_mat.shape[1]*100 # 4
                        loss = (loss + regloss) 
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                  
            num_seen_utts += num_utts
            if (batch_idx + 1) % accum_grad == 0:
                if rank == 0 and writer is not None:
                    writer.add_scalar('train_loss', loss, self.step)
                # Use mixed precision training
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    # Must invoke scaler.update() if unscale_() is used in the
                    # iteration to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # We don't check grad here since that if the gradient has
                    # inf/nan values, scaler.step will skip optimizer.step().
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                optimizer.zero_grad()
                if args['train_method'] != 'diags':
                    scheduler.step()
                self.step += 1
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                    batch_idx, num_total_batch,
                    loss.item() * accum_grad)
                if loss_att is not None:
                    log_str += 'loss_att {:.6f} '.format(loss_att.item())
                if loss_ctc is not None:
                    log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                if args['train_method'] == 'ctc_reg':
                    log_str += ' ctc_reg {:.3f}'.format(regloss)
                if args['train_method'] == 'diags':
                    log_str += ' dia_ratio {:.3f}'.format(dr)
                logging.debug(log_str)

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        num_total_batch = len(data_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, loss_ctc = model(feats, feats_lengths, target,
                                                 target_lengths)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        batch_idx, num_total_batch, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    logging.debug(log_str)

        return total_loss, num_seen_utts
