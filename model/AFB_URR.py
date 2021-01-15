import math
import torch
from torch import nn
from torch.nn import functional as NF
from torchvision.models import resnet50

import myutils


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(NF.relu(x))
        r = self.conv2(NF.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class EncoderM(nn.Module):
    def __init__(self, load_imagenet_params):
        super(EncoderM, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = resnet50(pretrained=load_imagenet_params)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) + self.conv1_m(in_m) + self.conv1_o(in_o)
        x = self.bn1(x)
        r1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(r1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024

        return r4, r1


class EncoderQ(nn.Module):
    def __init__(self, load_imagenet_params):
        super(EncoderQ, self).__init__()
        resnet = resnet50(pretrained=load_imagenet_params)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        r1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(r1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024

        return r4, r3, r2, r1


class KeyValue(nn.Module):

    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.keydim = keydim
        self.valdim = valdim
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        key = self.Key(x)
        key = key.view(*key.shape[:2], -1)  # obj_n, key_dim, pixel_n

        val = self.Value(x)
        val = val.view(*val.shape[:2], -1)  # obj_n, key_dim, pixel_n
        return key, val


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = 2

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + NF.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)

        return m


class Matcher(nn.Module):
    def __init__(self, thres_valid=1e-3, update_bank=False):
        super(Matcher, self).__init__()
        self.thres_valid = thres_valid
        self.update_bank = update_bank

    def forward(self, feature_bank, q_in, q_out):

        mem_out_list = []

        for i in range(0, feature_bank.obj_n):
            d_key, bank_n = feature_bank.keys[i].size()

            try:
                p = torch.matmul(feature_bank.keys[i].transpose(0, 1), q_in) / math.sqrt(d_key)  # THW, HW
                p = NF.softmax(p, dim=1)  # bs, bank_n, HW
                mem = torch.matmul(feature_bank.values[i], p)  # bs, D_o, HW
            except RuntimeError as e:
                device = feature_bank.keys[i].device
                key_cpu = feature_bank.keys[i].cpu()
                value_cpu = feature_bank.values[i].cpu()
                q_in_cpu = q_in.cpu()

                p = torch.matmul(key_cpu.transpose(0, 1), q_in_cpu) / math.sqrt(d_key)  # THW, HW
                p = NF.softmax(p, dim=1)  # bs, bank_n, HW
                mem = torch.matmul(value_cpu, p).to(device)  # bs, D_o, HW
                p = p.to(device)
                print('\tLine 158. GPU out of memory, use CPU', f'p size: {p.shape}')

            mem_out_list.append(torch.cat([mem, q_out], dim=1))

            if self.update_bank:
                try:
                    ones = torch.ones_like(p)
                    zeros = torch.zeros_like(p)
                    bank_cnt = torch.where(p > self.thres_valid, ones, zeros).sum(dim=2)[0]
                except RuntimeError as e:
                    device = p.device
                    p = p.cpu()
                    ones = torch.ones_like(p)
                    zeros = torch.zeros_like(p)
                    bank_cnt = torch.where(p > self.thres_valid, ones, zeros).sum(dim=2)[0].to(device)
                    print('\tLine 170. GPU out of memory, use CPU', f'p size: {p.shape}')

                feature_bank.info[i][:, 1] += torch.log(bank_cnt + 1)

        mem_out_tensor = torch.stack(mem_out_list, dim=0).transpose(0, 1)  # bs, obj_n, dim, pixel_n

        return mem_out_tensor


class Decoder(nn.Module):
    def __init__(self, device):  # mdim_global = 256
        super(Decoder, self).__init__()

        self.device = device
        mdim_global = 256
        mdim_local = 32
        local_size = 7

        # Patch-wise
        self.convFM = nn.Conv2d(1024, mdim_global, kernel_size=3, padding=1, stride=1)
        self.ResMM = ResBlock(mdim_global, mdim_global)
        self.RF3 = Refine(512, mdim_global)  # 1/8 -> 1/8
        self.RF2 = Refine(256, mdim_global)  # 1/8 -> 1/4
        self.pred2 = nn.Conv2d(mdim_global, 2, kernel_size=3, padding=1, stride=1)

        # Local
        self.local_avg = nn.AvgPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_max = nn.MaxPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_convFM = nn.Conv2d(128, mdim_local, kernel_size=3, padding=1, stride=1)
        self.local_ResMM = ResBlock(mdim_local, mdim_local)
        self.local_pred2 = nn.Conv2d(mdim_local, 2, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, patch_match, r3, r2, r1=None, feature_shape=None):
        p = self.ResMM(self.convFM(patch_match))
        p = self.RF3(r3, p)  # out: 1/8, 256
        p = self.RF2(r2, p)  # out: 1/4, 256
        p = self.pred2(NF.relu(p))

        p = NF.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)

        bs, obj_n, h, w = feature_shape
        rough_seg = NF.softmax(p, dim=1)[:, 1]
        rough_seg = rough_seg.view(bs, obj_n, h, w)
        rough_seg = NF.softmax(rough_seg, dim=1)  # object-level normalization

        # Local refinement
        uncertainty = myutils.calc_uncertainty(rough_seg)
        uncertainty = uncertainty.expand(-1, obj_n, -1, -1).reshape(bs * obj_n, 1, h, w)

        rough_seg = rough_seg.view(bs * obj_n, 1, h, w)  # bs*obj_n, 1, h, w
        r1_weighted = r1 * rough_seg
        r1_local = self.local_avg(r1_weighted)  # bs*obj_n, 64, h, w
        r1_local = r1_local / (self.local_avg(rough_seg) + 1e-8)  # neighborhood reference
        r1_conf = self.local_max(rough_seg)  # bs*obj_n, 1, h, w

        local_match = torch.cat([r1, r1_local], dim=1)
        q = self.local_ResMM(self.local_convFM(local_match))
        q = r1_conf * self.local_pred2(NF.relu(q))

        p = p + uncertainty * q
        p = NF.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)
        p = NF.softmax(p, dim=1)[:, 1]  # no, h, w

        return p


class AFB_URR(nn.Module):
    def __init__(self, device, update_bank, load_imagenet_params=False):
        super(AFB_URR, self).__init__()

        self.device = device
        self.encoder_m = EncoderM(load_imagenet_params)
        self.encoder_q = EncoderQ(load_imagenet_params)

        self.keyval_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.global_matcher = Matcher(update_bank=update_bank)
        self.decoder = Decoder(device)

    def memorize(self, frame, mask):

        _, K, H, W = mask.shape

        (frame, mask), pad = myutils.pad_divide_by([frame, mask], 16, (frame.size()[2], frame.size()[3]))

        frame = frame.expand(K, -1, -1, -1)  # obj_n, 3, h, w
        mask = mask[0].unsqueeze(1).float()
        mask_ones = torch.ones_like(mask)
        mask_inv = (mask_ones - mask).clamp(0, 1)

        r4, r1 = self.encoder_m(frame, mask, mask_inv)

        k4, v4 = self.keyval_r4(r4)  # num_objects, 128 and 512, H/16, W/16
        k4_list = [k4[i] for i in range(K)]
        v4_list = [v4[i] for i in range(K)]

        return k4_list, v4_list

    def segment(self, frame, fb_global):

        obj_n = fb_global.obj_n

        if not self.training:
            [frame], pad = myutils.pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, r1 = self.encoder_q(frame)
        bs, _, global_match_h, global_match_w = r4.shape
        _, _, local_match_h, local_match_w = r1.shape

        k4, v4 = self.keyval_r4(r4)  # 1, dim, H/16, W/16
        res_global = self.global_matcher(fb_global, k4, v4)
        res_global = res_global.reshape(bs * obj_n, v4.shape[1] * 2, global_match_h, global_match_w)

        r3_size = r3.shape
        r2_size = r2.shape
        r3 = r3.unsqueeze(1).expand(-1, obj_n, -1, -1, -1).reshape(bs * obj_n, *r3_size[1:])
        r2 = r2.unsqueeze(1).expand(-1, obj_n, -1, -1, -1).reshape(bs * obj_n, *r2_size[1:])

        r1_size = r1.shape
        r1 = r1.unsqueeze(1).expand(-1, obj_n, -1, -1, -1).reshape(bs * obj_n, *r1_size[1:])
        feature_size = (bs, obj_n, r1_size[2], r1_size[3])
        score = self.decoder(res_global, r3, r2, r1, feature_size)

        score = score.view(obj_n, bs, *frame.shape[-2:]).permute(1, 0, 2, 3)

        if self.training:
            uncertainty = myutils.calc_uncertainty(NF.softmax(score, dim=1))
            uncertainty = uncertainty.view(bs, -1).norm(p=2, dim=1) / math.sqrt(frame.shape[-2] * frame.shape[-1])  # [B,1,H,W]
            uncertainty = uncertainty.mean()
        else:
            uncertainty = None

        score = torch.clamp(score, 1e-7, 1 - 1e-7)
        score = torch.log((score / (1 - score)))

        if not self.training:
            if pad[2] + pad[3] > 0:
                score = score[:, :, pad[2]:-pad[3], :]
            if pad[0] + pad[1] > 0:
                score = score[:, :, :, pad[0]:-pad[1]]

        return score, uncertainty

    def forward(self, x):
        pass
