import numpy as np
from tqdm import tqdm, trange
import os
import argparse
from PIL import Image
import torch
from torch import utils
from torch.nn import functional as F

from dataset import DAVIS_Test_DS, YouTube_Test_DS, LongVideo_Test_DS, Video_DS
from model import AFB_URR, FeatureBank
import myutils

torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser(description='Eval AFB-URR')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU card id.')
    parser.add_argument('--level', type=int, default=1, required=True,
                        help='1: DAVIS17. 2: Youtube-VOS. 3: LongVideo')
    parser.add_argument('--budget', type=int, default='300000',
                        help='Max number of features that feature bank can store. Default: 300000')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize data.')
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='Dataset folder.')
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--prefix', type=str,
                        help='Prefix to the model name.')
    parser.add_argument('--update-rate', type=float, default=0.1,
                        help='Update Rate. Impact of merging new features.')
    parser.add_argument('--merge-thres', type=float, default=0.95,
                        help='Merging Rate. If similarity higher than this, then merge, else append.')
    return parser.parse_args()


def eval_DAVIS(model, model_name, dataloader):
    fps = myutils.FrameSecondMeter()

    for seq_idx, V in enumerate(dataloader):

        frames, masks, obj_n, info = V
        seq_name = info['name'][0]
        obj_n = obj_n.item()

        seg_dir = os.path.join('./output', model_name, seq_name)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        if args.viz:
            overlay_dir = os.path.join('./overlay', model_name, seq_name)
            if not os.path.exists(overlay_dir):
                os.makedirs(overlay_dir)

        frames, masks = frames[0].to(device), masks[0].to(device)
        frame_n = info['num_frames'][0].item()

        pred_mask = masks[0:1]
        pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
        seg_path = os.path.join(seg_dir, '00000.png')
        myutils.save_seg_mask(pred, seg_path, palette)

        if args.viz:
            overlay_path = os.path.join(overlay_dir, '00000.png')
            myutils.save_overlay(frames[0], pred, overlay_path, palette)

        fb = FeatureBank(obj_n, args.budget, device, update_rate=args.update_rate, thres_close=args.merge_thres)
        k4_list, v4_list = model.memorize(frames[0:1], pred_mask)
        fb.init_bank(k4_list, v4_list)

        for t in tqdm(range(1, frame_n), desc=f'{seq_idx} {seq_name}'):

            score, _ = model.segment(frames[t:t + 1], fb)

            pred_mask = F.softmax(score, dim=1)

            pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
            seg_path = os.path.join(seg_dir, f'{t:05d}.png')
            myutils.save_seg_mask(pred, seg_path, palette)

            if t < frame_n - 1:
                k4_list, v4_list = model.memorize(frames[t:t + 1], pred_mask)
                fb.update(k4_list, v4_list, t)

            if args.viz:
                overlay_path = os.path.join(overlay_dir, f'{t:05d}.png')
                myutils.save_overlay(frames[t], pred, overlay_path, palette)

        fps.add_frame_n(frame_n)

    fps.end()
    print(myutils.gct(), 'fps:', fps.fps)


def eval_YouTube(model, model_name, dataloader):
    seq_n = len(dataloader)

    for seq_idx, V in enumerate(dataloader):

        frames, masks, obj_n, info = V

        frames, masks = frames[0].to(device), masks[0].to(device)
        frame_n = info['num_frames'][0].item()
        seq_name = info['name'][0]
        obj_n = obj_n.item()
        obj_st = [info['obj_st'][0, i].item() for i in range(obj_n)]
        basename_list = [info['basename_list'][i][0] for i in range(frame_n)]
        basename_to_save = [info['basename_to_save'][i][0] for i in range(len(info['basename_to_save']))]
        obj_vis = info['obj_vis'][0]
        original_size = (info['original_size'][0].item(), info['original_size'][1].item())

        seg_dir = os.path.join('./output', model_name, seq_name)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        if args.viz:
            overlay_dir = os.path.join('./overlay', model_name, seq_name)
            if not os.path.exists(overlay_dir):
                os.makedirs(overlay_dir)

        # Compose the first mask
        pred_mask = torch.zeros_like(masks).unsqueeze(0).float()
        for i in range(1, obj_n):
            if obj_st[i] == 0:
                pred_mask[0, i] = masks[i]
        pred_mask[0, 0] = 1 - pred_mask.sum(dim=1)

        pred_mask_output = F.interpolate(pred_mask, original_size)
        pred = torch.argmax(pred_mask_output[0], dim=0).cpu().numpy().astype(np.uint8)
        seg_path = os.path.join(seg_dir, basename_list[0] + '.png')
        myutils.save_seg_mask(pred, seg_path, palette)

        if args.viz:
            frame_out = F.interpolate(frames[0].unsqueeze(0), original_size).squeeze(0)
            overlay_path = os.path.join(overlay_dir, basename_list[0] + '.png')
            myutils.save_overlay(frame_out, pred, overlay_path, palette)

        fb = FeatureBank(obj_n, args.budget, device, update_rate=args.update_rate,
                         thres_close=args.merge_thres)  # 0.1 0.95

        k4_list, v4_list = model.memorize(frames[0:1], pred_mask)
        fb.init_bank(k4_list, v4_list)

        for t in trange(1, frame_n, desc=f'{seq_idx:3d}/{seq_n:3d} {seq_name}'):

            score, _ = model.segment(frames[t:t + 1], fb)

            reset_list = list()
            for i in range(1, obj_n):
                # If this object is invisible.
                if obj_vis[t, i] == 0:
                    score[0, i] = -1000

                # If this object appears, reset the score map
                if obj_st[i] == t:
                    reset_list.append(i)
                    score[0, i] = -1000
                    score[0, i][masks[i]] = 1000
                    for j in range(obj_n):
                        if j != i:
                            score[0, j][masks[i]] = -1000

            pred_mask = F.softmax(score, dim=1)

            if t < frame_n - 1:
                k4_list, v4_list = model.memorize(frames[t:t + 1], pred_mask)
                if len(reset_list) > 0:
                    fb.init_bank(k4_list, v4_list, t)
                else:
                    fb.update(k4_list, v4_list, t)

            if basename_list[t] in basename_to_save:
                pred_mask_output = F.interpolate(score, original_size)
                pred = torch.argmax(pred_mask_output[0], dim=0).cpu().numpy().astype(np.uint8)
                seg_path = os.path.join(seg_dir, basename_list[t] + '.png')
                myutils.save_seg_mask(pred, seg_path, palette)

                if args.viz:
                    frame_out = F.interpolate(frames[t].unsqueeze(0), original_size).squeeze(0)
                    overlay_path = os.path.join(overlay_dir, basename_list[t] + '.png')
                    myutils.save_overlay(frame_out, pred, overlay_path, palette)

        fb.print_peak_mem()


def eval_LongVideo(model, model_name, dataloader):
    for seq_idx, V in enumerate(dataloader):

        video_name, img_dir, mask_dir = V
        seq_name = video_name[0]

        seq_dataset = Video_DS(img_dir[0], mask_dir[0])
        seq_loader = utils.data.DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=2)

        seg_dir = os.path.join('./output', model_name, seq_name)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        if args.viz:
            overlay_dir = os.path.join('./overlay', model_name, seq_name)
            if not os.path.exists(overlay_dir):
                os.makedirs(overlay_dir)

        first_frame = seq_dataset.first_frame.unsqueeze(0).to(device)
        first_mask = seq_dataset.first_mask.unsqueeze(0).to(device)
        frame_n = seq_dataset.video_len
        obj_n = seq_dataset.obj_n

        pred_mask = first_mask
        pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
        seg_path = os.path.join(seg_dir, f'{seq_dataset.first_name}.png')
        myutils.save_seg_mask(pred, seg_path, palette)

        if args.viz:
            overlay_path = os.path.join(overlay_dir, f'{seq_dataset.first_name}.png')
            myutils.save_overlay(first_frame[0], pred, overlay_path, palette)

        fb = FeatureBank(obj_n, args.budget, device, update_rate=args.update_rate, thres_close=args.merge_thres)

        k4_list, v4_list = model.memorize(first_frame, pred_mask)
        fb.init_bank(k4_list, v4_list)

        for idx, (frame, frame_name) in enumerate(tqdm(seq_loader, desc=f'{seq_idx} {seq_name}')):

            t = idx + 1

            frame = frame.to(device)
            score, _ = model.segment(frame, fb)

            pred_mask = F.softmax(score, dim=1)
            pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
            seg_path = os.path.join(seg_dir, f'{frame_name[0]}.png')
            myutils.save_seg_mask(pred, seg_path, palette)

            if t < frame_n - 1:
                k4_list, v4_list = model.memorize(frame, pred_mask)
                fb.update(k4_list, v4_list, t)

            if args.viz:
                overlay_path = os.path.join(overlay_dir, f'{frame_name[0]}.png')
                myutils.save_overlay(frame[0], pred, overlay_path, palette)

        fb.print_peak_mem()


def main():
    model = AFB_URR(device, update_bank=True, load_imagenet_params=False)
    model = model.to(device)
    model.eval()

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            end_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'], strict=False)
            train_loss = checkpoint['loss']
            seed = checkpoint['seed']
            print(myutils.gct(),
                  f'Loaded checkpoint {args.resume}. (end_epoch: {end_epoch}, train_loss: {train_loss}, seed: {seed})')
        else:
            print(myutils.gct(), f'No checkpoint found at {args.resume}')
            raise IOError

    if args.level == 1:
        model_name = 'AFB-URR_DAVIS_17val'
        dataset = DAVIS_Test_DS(args.dataset, '2017/val.txt')
    elif args.level == 2:
        model_name = 'AFB-URR_YoutubeVOS'
        dataset = YouTube_Test_DS(args.dataset)
    elif args.level == 3:
        model_name = 'AFB-URR_LongVideo'
        dataset = LongVideo_Test_DS(args.dataset, 'val.txt')
    else:
        raise ValueError(f'{args.level} is unknown.')

    if args.prefix:
        model_name += f'_{args.prefix}'
    dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print(myutils.gct(), f'Model name: {model_name}')

    if args.level == 1:
        eval_DAVIS(model, model_name, dataloader)
    elif args.level == 2:
        eval_YouTube(model, model_name, dataloader)
    elif args.level == 3:
        eval_LongVideo(model, model_name, dataloader)


if __name__ == '__main__':

    args = get_args()
    print(myutils.gct(), 'Args =', args)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
    else:
        raise ValueError('CUDA is required. --gpu must be >= 0.')

    palette = Image.open(os.path.join(args.dataset, 'mask_palette.png')).getpalette()

    main()

    print(myutils.gct(), 'Evaluation done.')
