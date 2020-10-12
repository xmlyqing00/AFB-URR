import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.utils import data
from tensorboardX import SummaryWriter

from dataset import PreTrain_DS, DAVIS_Train_DS, YouTube_Train_DS
from model import AFB_URR, FeatureBank
import myutils


def get_args():
    parser = argparse.ArgumentParser(description='Train AFB-URR')
    parser.add_argument('--gpu', type=int, help='0, 1, 2', default=0)
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='Dataset folder.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed.')
    parser.add_argument('--log', action='store_true',
                        help='Save the training results.')
    parser.add_argument('--level', type=int, default=0,
                        help='0: pretrain. 1: DAVIS. 2: Youtube-VOS.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate, default 1e-5.')
    parser.add_argument('--lu', type=float, default=0.5,
                        help='Regularization rate, default 0.5.')
    parser.add_argument('--resume', type=str,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--new', action='store_true',
                        help='Train the model from the begining.')
    parser.add_argument('--scheduler-step', type=int, default=25,
                        help='Scheduler step size. Default 1.')
    parser.add_argument('--total-epochs', type=int, default=100,
                        help='Total running epochs. Default 200.')
    parser.add_argument('--budget', type=int, default=200000)
    parser.add_argument('--obj-n', type=int, default=3)
    parser.add_argument('--clip-n', type=int, default=6)  # 6

    return parser.parse_args()


def run_pretrain(model, dataloader, criterion, optimizer):
    stats = myutils.AvgMeter()

    progress_bar = tqdm(dataloader, desc='Pre Train')
    for iter_idx, sample in enumerate(progress_bar):
        frames, masks, obj_n, info = sample

        if obj_n.item() == 1:
            continue

        frames, masks = frames[0].to(device), masks[0].to(device)

        # with torch.autograd.detect_anomaly():

        k4, v4 = model.memorize(frames[0:1], masks[0:1])
        scores = model.segment(frames[1:], k4, v4)
        label = torch.argmax(masks[1:], dim=1).long()

        optimizer.zero_grad()
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()

        stats.update(loss.item())
        progress_bar.set_postfix(loss=('%.6f' % stats.avg))
        progress_bar.update(1)

        # Save tmp model
        if iter_idx == 40000 or iter_idx == 80000:
            checkpoint = {
                'epoch': iter_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': stats.avg,
                'seed': -1,
            }

            cp_path = f'tmp/cp_{iter_idx}.pth'
            torch.save(checkpoint, cp_path)
            print('Save to', cp_path)

    progress_bar.close()

    return stats.avg


def run_maintrain(model, dataloader, criterion, optimizer):

    stats = myutils.AvgMeter()
    uncertainty_stats = myutils.AvgMeter()

    progress_bar = tqdm(dataloader, desc='Main Train')
    for iter_idx, sample in enumerate(progress_bar):
        frames, masks, obj_n, info = sample

        obj_n = obj_n.item()
        if obj_n == 1:
            continue

        frames, masks = frames[0].to(device), masks[0].to(device)

        fb_global = FeatureBank(obj_n, args.budget, device)
        k4_list, v4_list = model.memorize(frames[0:1], masks[0:1])
        fb_global.init_bank(k4_list, v4_list)

        scores, uncertainty = model.segment(frames[1:], fb_global)
        label = torch.argmax(masks[1:], dim=1).long()

        optimizer.zero_grad()
        loss = criterion(scores, label)
        loss = loss + args.lu * uncertainty
        loss.backward()
        optimizer.step()

        uncertainty_stats.update(uncertainty.item())
        stats.update(loss.item())
        progress_bar.set_postfix(loss=f'{loss.item():.5f} ({stats.avg:.5f} {uncertainty_stats.avg:.5f})')

        # For debug
        # print(info)
        # myutils.vis_result(frames, masks, scores)

        # Save tmp model
        if iter_idx == 40000 or iter_idx == 80000 or iter_idx == 130000:
            checkpoint = {
                'epoch': iter_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': stats.avg,
                'seed': -1,
            }

            cp_path = f'tmp/cp_{iter_idx}.pth'
            torch.save(checkpoint, cp_path)
            print('Save to', cp_path)

    progress_bar.close()

    return stats.avg


def main():
    # torch.autograd.set_detect_anomaly(True)

    if args.level == 0:
        dataset = PreTrain_DS(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
    elif args.level == 1:
        dataset = DAVIS_Train_DS(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
    elif args.level == 2:
        dataset = YouTube_Train_DS(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
    else:
        raise ValueError(f'{args.level} is unknown.')

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    print(myutils.gct(), f'Load level {args.level} dataset: {len(dataset)} training cases.')

    model = AFB_URR(device, update_bank=False, load_imagenet_params=True)
    model = model.to(device)
    model.train()
    model.apply(myutils.set_bn_eval)  # turn-off BN

    params = model.parameters()
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, params), args.lr)

    start_epoch = 0
    best_loss = 100000000
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'], strict=False)
            seed = checkpoint['seed']

            if not args.new:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_loss = checkpoint['loss']
                print(myutils.gct(),
                      f'Loaded checkpoint {args.resume} (epoch: {start_epoch-1}, best loss: {best_loss})')
            else:
                if args.seed < 0:
                    seed = int(time.time())
                else:
                    seed = args.seed
                print(myutils.gct(), f'Loaded checkpoint {args.resume}. Train from the beginning.')
        else:
            print(myutils.gct(), f'No checkpoint found at {args.resume}')
            raise IOError
    else:

        if args.seed < 0:
            seed = int(time.time())
        else:
            seed = args.seed

    print(myutils.gct(), 'Random seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.5, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.total_epochs):

        lr = scheduler.get_last_lr()[0]
        print('')
        print(myutils.gct(), f'Epoch: {epoch} lr: {lr}')

        loss = run_maintrain(model, dataloader, criterion, optimizer)
        if args.log:

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'seed': seed,
            }

            checkpoint_path = f'{model_path}/final.pth'
            torch.save(checkpoint, checkpoint_path)

            if best_loss > loss:
                best_loss = loss

                checkpoint_path = f'{model_path}/epoch_{epoch:03d}_loss_{loss:.03f}.pth'
                torch.save(checkpoint, checkpoint_path)

                checkpoint_path = f'{model_path}/best.pth'
                torch.save(checkpoint, checkpoint_path)

                print('Best model updated.')

        scheduler.step()


if __name__ == '__main__':

    # Pretrain
    # --level 0 --scheduler-step 1 --total-epoch 5
    # DAVIS 17
    # --log --level 1 --lr 1e-5 --clip-n 3 --new --resume logs/level0_20200505-141005/model/epoch_003_loss_0.020.pth
    # --level 1 --lr 1e-5 --local --new --resume logs/level0_20200505-141005/model/epoch_003_loss_0.020.pth
    # DAVIS17 + entire
    # --local --level 1 --new --resume logs/level0_20200505-141005/model/epoch_003_loss_0.020.pth --log --clip-n 4 --lr 5e-6

    args = get_args()
    print(myutils.gct(), f'Args = {args}')

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
    else:
        raise ValueError('CUDA is required. --gpu must be >= 0.')

    if args.log:
        if not os.path.exists('logs'):
            os.makedirs('logs')

        prefix = f'level{args.level}'
        log_dir = 'logs/{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S'))
        log_path = os.path.join(log_dir, 'log')
        model_path = os.path.join(log_dir, 'model')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        myutils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(log_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('myutils/*.py', recursive=True))

        vis_writer = SummaryWriter(log_path)
        vis_writer_step = 0

        print(myutils.gct(), f'Create log dir: {log_dir}')

    main()

    if args.log:
        vis_writer.close()

    print(myutils.gct(), 'Training done.')
