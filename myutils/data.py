import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
import cv2

import torch
from torch.nn import functional as NF
from torchvision.transforms import functional as TF


def calc_uncertainty(score):

    # seg shape: bs, obj_n, h, w
    score_top, _ = score.topk(k=2, dim=1)
    uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
    uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
    return uncertainty


def parse_sample(sample, device):
    sample['img'] = sample['img'].to(device)
    sample['label'] = sample['label'].to(device=device, non_blocking=True)
    return sample


def vis_sample(sample, name, tmp_dir='./tmp/'):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    img = TF.to_pil_image(sample['img'] * std + mean)
    label = TF.to_pil_image(sample['label'])
    img.save(os.path.join(tmp_dir, f'{name}_img.png'))
    label.save(os.path.join(tmp_dir, f'{name}_label.png'))


def vis_result(frames, masks, scores, tmp_dir='./tmp/'):
    palette = Image.open(os.path.join('assets/mask_palette.png')).getpalette()
    for i in range(frames.shape[0]):
        img = TF.to_pil_image(frames[i].cpu())
        img.save(os.path.join(tmp_dir, f'{i}_img.png'))

        label = torch.argmax(masks[i], dim=0)
        label = TF.to_pil_image(label.cpu().int()).convert('P')
        label.putpalette(palette)
        label.save(os.path.join(tmp_dir, f'{i}_label.png'))

        if i > 0:
            score = torch.argmax(scores[i-1], dim=0)
            score = TF.to_pil_image(score.detach().cpu().int()).convert('P')
            score.putpalette(palette)
            score.save(os.path.join(tmp_dir, f'{i}_score.png'))


def save_seg(seg, obj_labels, save_path):
    img = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for idx, label_id in enumerate(obj_labels):
        img[(seg == idx)] = label_id[::-1]  # RGB -> BGR

    cv2.imwrite(save_path, img)


def unify_features(features):
    output_size = features['f0'].shape[-2:]
    feature_tuple = tuple()

    for key, f in features.items():
        if key != 'f0':
            f = NF.interpolate(
                f,
                size=output_size, mode='bilinear', align_corners=False
            )
        feature_tuple += (f,)

    unified_feature = torch.cat(feature_tuple, dim=1)

    return unified_feature


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(NF.pad(inp, pad_array))

    return out_list, pad_array


def save_seg_mask(pred, seg_path, palette):

    seg_img = Image.fromarray(pred)
    seg_img.putpalette(palette)
    seg_img.save(seg_path)


def add_overlay(img, mask, colors, alpha=0.7, cscale=1):

    ids = np.unique(mask)
    img_overlay = img.copy()
    ones_np = np.ones(img.shape) * (1 - alpha)

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    for i in ids[1:]:

        canvas = img * alpha + ones_np * np.array(colors[i])[::-1]

        binary_mask = mask == i
        img_overlay[binary_mask] = canvas[binary_mask]

        contour = binary_dilation(binary_mask) ^ binary_mask
        img_overlay[contour, :] = 0

    return img_overlay


def save_overlay(img, mask, overlay_path, colors=[255, 0, 0], alpha=0.4, cscale=1):

    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_overlay = add_overlay(img, mask, colors, alpha, cscale)
    cv2.imwrite(overlay_path, img_overlay)


