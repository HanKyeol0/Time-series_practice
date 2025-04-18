import wandb
import time
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from accelerate import Accelerator
from accelerate.logging import get_logger

from utils.metrics import cal_metric
from utils.utils import Float32Encoder
from utils.tools import EarlyStopping, adjust_learning_rate, check_forecasting_graph

_logger = get_logger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def training_dl(
    model, trainloader, validloader, criterion, optimizer, accelerator: Accelerator, 
    epochs: int, eval_epochs: int, log_epochs: int, log_eval_iter: int, wandb_iter: int,
    use_wandb: bool, ckp_metric: str, savedir: str, model_name: str, 
    pred_len: int, label_len: int, early_stopping_metric: str, early_stopping_count: int,
    lradj: int, learning_rate: int, model_config: dict):
    
    # average meter
    batch_time_m = AverageMeter() # the role of AverageMeter is to calculate the average of the values
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # set mode
    model.train()
    optimizer.zero_grad()
    end_time = time.time()
    
    early_stopping = EarlyStopping(patience=early_stopping_count)
    
    # init best score and step
    best_score = np.inf
    wandb_iteration = 0
    
    _logger.info(f"\n 🔹 Training started")

    for epoch in range(epochs):
        epoch_time = time.time()
        for idx, (item) in enumerate(trainloader):
            data_time_m.update(time.time() - end_time)

            input_ts = item['x'].to(accelerator.device)
            target_ts = item['y'].to(accelerator.device)
            input_ts_mark = item['x_ts'].to(accelerator.device)
            target_ts_mark = item['y_ts'].to(accelerator.device)

            x_dec = torch.zeros_like(target_ts[:, :pred_len, :]).to(accelerator.device)
            x_dec = torch.cat([target_ts[:, -label_len:, :], x_dec], dim=1)
            x_dec_ts = torch.zeros_like(target_ts_mark[:, :pred_len, :]).to(accelerator.device)
            x_dec_ts = torch.cat([target_ts_mark[:, -label_len:, :], x_dec_ts], dim=1)

            outputs = model(input_ts, input_ts_mark, x_dec, x_dec_ts)      
            loss = criterion(outputs, target_ts)
            
            accelerator.backward(loss)
            
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            
            losses_m.update(loss.item(), n = target_ts.size(0))
            
            # batch time
            batch_time_m.update(time.time() - end_time)
            wandb_iteration += 1
            
            if use_wandb and (wandb_iteration+1) % wandb_iter:
                train_results = OrderedDict([
                    ('lr',optimizer.param_groups[0]['lr']),
                    ('train_loss',losses_m.avg)
                ])
                wandb.log(train_results, step=wandb_iteration+1)
        
        if (epoch+1) % log_epochs == 0:
            _logger.info('EPOCH {:>3d}/{} | TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (epoch+1), epochs, 
                        (idx+1), 
                        len(trainloader), 
                        loss       = losses_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = input_ts.size(0) / batch_time_m.val,
                        rate_avg   = input_ts.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
                    
        if (epoch+1) % eval_epochs == 0:
            eval_metrics = test_dl(
                accelerator   = accelerator,
                model         = model, 
                dataloader    = validloader, 
                criterion     = criterion,
                name          = 'VALID',
                log_interval  = log_eval_iter,
                label_len     = label_len,
                pred_len      = pred_len,
                return_output = False,
                savedir       = savedir,
                model_name    = model_name,
                model_config  = model_config
                )

            model.train()
            
            # eval results
            eval_results = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
            
            # wandb
            if use_wandb:
                wandb.log(eval_results, step=wandb_iteration+1)
                
            # check_point
            if best_score > eval_metrics[ckp_metric]:
                # save results
                state = {'best_epoch':epoch ,
                            'best_step':idx+1, 
                            f'best_{ckp_metric}':eval_metrics[ckp_metric]}
                
                print('Save best model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
                    to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))
                
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    state.update(eval_results)
                    json.dump(state, open(os.path.join(savedir, f'forecasting_best_results.json'),'w'), 
                                indent='\t', cls=Float32Encoder)

                # save model
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best {0} {1:6.6f} to {2:6.6f}'.format(ckp_metric.upper(), best_score, eval_metrics[ckp_metric]))
                    _logger.info("\n✅ Saved best model")

                best_score = eval_metrics[ckp_metric]
                
            early_stopping(eval_metrics[early_stopping_metric])
            if early_stopping.early_stop:
                _logger.info("⏳ Early stopping triggered")
                break
        
        adjust_learning_rate(optimizer, epoch + 1, lradj, learning_rate)

        end_time = time.time()

    # logging best score and step
    _logger.info('Best Metric: {0:6.6f} (step {1:})\n'.format(state[f'best_{ckp_metric}'], state['best_step']))
    
    # save latest model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

        print('Save latest model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
            to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))

        # save latest results
        state = {'best_epoch':epoch ,'best_step':idx+1, f'latest_{ckp_metric}':eval_metrics[ckp_metric]}
        state.update(eval_results)
        json.dump(state, open(os.path.join(savedir, f'forecasting_latest_results.json'),'w'), indent='\t', cls=Float32Encoder)
    _logger.info("\n🎉 Training complete for all datasets")

def test_dl(model, dataloader, criterion, accelerator: Accelerator, 
            log_interval: int, pred_len: int, label_len: int, savedir: str, model_config: dict,
            model_name: str, name: str = 'TEST', return_output: bool = False) -> dict:
    
    _logger.info(f'\n[🔍 Start {name} Evaluation]')
    
    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # targets and outputs
    total_targets = []
    total_outputs = []
    
    end_time = time.time()
    
    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            input_ts = item['x'].to(accelerator.device)
            target_ts = item['y'].to(accelerator.device)
            input_ts_mark = item['x_ts'].to(accelerator.device)
            target_ts_mark = item['y_ts'].to(accelerator.device)

            x_dec = torch.zeros_like(target_ts[:, -label_len:, :]).to(accelerator.device)
            x_dec = torch.cat([target_ts[:, -label_len:, :], x_dec], dim=1)
            x_dec_ts = torch.zeros_like(target_ts_mark[:, -label_len:, :]).to(accelerator.device)
            x_dec_ts = torch.cat([target_ts_mark[:, -label_len:, :], x_dec_ts], dim=1)

            outputs = model(input_ts, input_ts_mark, x_dec, x_dec_ts)      
            loss = criterion(outputs, target_ts)
                
            loss = accelerator.gather(loss)
            loss = torch.mean(loss)
            
            outputs, target_ts = accelerator.gather_for_metrics((outputs.contiguous(), target_ts.contiguous()))

            losses_m.update(loss.item(), n=input_ts.size(0))
            outputs = outputs.detach().cpu().numpy()
            target_ts = target_ts.detach().cpu().numpy()

            total_outputs.append(outputs)
            total_targets.append(target_ts)
            
            # batch time        
            batch_time_m.update(time.time() - end_time)
            
            if (idx+1) % log_interval == 0:
                _logger.info('{name} [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(dataloader),
                            name       = name, 
                            loss       = losses_m, 
                            batch_time = batch_time_m,
                            rate       = input_ts.size(0) / batch_time_m.val,
                            rate_avg   = input_ts.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))
                
            end_time = time.time()
    
    total_outputs = np.concatenate(total_outputs, axis=0)
    total_targets = np.concatenate(total_targets, axis=0)

    results = cal_metric(
        pred=total_outputs,
        true=total_targets,
    )

    return results