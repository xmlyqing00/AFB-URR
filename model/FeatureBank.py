import numpy as np
import torch
import torch.nn.functional as NF

from torch_scatter import scatter_mean


class FeatureBank:

    def __init__(self, obj_n, memory_budget, device, update_rate=0.1, thres_close=0.95):
        self.obj_n = obj_n
        self.update_rate = update_rate
        self.thres_close = thres_close
        self.device = device

        self.info = [None for _ in range(obj_n)]
        self.peak_n = np.zeros(obj_n)
        self.replace_n = np.zeros(obj_n)

        self.class_budget = memory_budget // obj_n
        if obj_n == 2:
            self.class_budget = 0.8 * self.class_budget

    def init_bank(self, keys, values, frame_idx=0):

        self.keys = keys
        self.values = values

        for class_idx in range(self.obj_n):
            _, bank_n = keys[class_idx].shape
            self.info[class_idx] = torch.zeros((bank_n, 2), device=self.device)
            self.info[class_idx][:, 0] = frame_idx
            self.peak_n[class_idx] = max(self.peak_n[class_idx], self.info[class_idx].shape[0])

    def update(self, prev_key, prev_value, frame_idx):

        for class_idx in range(self.obj_n):

            d_key, bank_n = self.keys[class_idx].shape
            d_val, _ = self.values[class_idx].shape

            normed_keys = NF.normalize(self.keys[class_idx], dim=0)
            normed_prev_key = NF.normalize(prev_key[class_idx], dim=0)
            mag_keys = self.keys[class_idx].norm(p=2, dim=0)
            corr = torch.mm(normed_keys.transpose(0, 1), normed_prev_key)  # bank_n, prev_n
            related_bank_idx = corr.argmax(dim=0, keepdim=True)  # 1, HW
            related_bank_corr = torch.gather(corr, 0, related_bank_idx)  # 1, HW

            # greater than threshold, merge them
            selected_idx = (related_bank_corr[0] > self.thres_close).nonzero()
            class_related_bank_idx = related_bank_idx[0, selected_idx[:, 0]]  # selected_HW
            unique_related_bank_idx, cnt = class_related_bank_idx.unique(dim=0, return_counts=True)  # selected_HW

            # Update key
            key_bank_update = torch.zeros((d_key, bank_n), dtype=torch.float).cuda()  # d_key, THW
            key_bank_idx = class_related_bank_idx.unsqueeze(0).expand(d_key, -1)  # d_key, HW
            scatter_mean(normed_prev_key[:, selected_idx[:, 0]], key_bank_idx, dim=1, out=key_bank_update)
            # d_key, selected_HW

            self.keys[class_idx][:, unique_related_bank_idx] = \
                mag_keys[unique_related_bank_idx] * \
                ((1 - self.update_rate) * normed_keys[:, unique_related_bank_idx] + \
                 self.update_rate * key_bank_update[:, unique_related_bank_idx])

            # Update value
            normed_values = NF.normalize(self.values[class_idx], dim=0)
            normed_prev_value = NF.normalize(prev_value[class_idx], dim=0)
            mag_values = self.values[class_idx].norm(p=2, dim=0)
            val_bank_update = torch.zeros((d_val, bank_n), dtype=torch.float).cuda()
            val_bank_idx = class_related_bank_idx.unsqueeze(0).expand(d_val, -1)
            scatter_mean(normed_prev_value[:, selected_idx[:, 0]], val_bank_idx, dim=1, out=val_bank_update)

            self.values[class_idx][:, unique_related_bank_idx] = \
                mag_values[unique_related_bank_idx] * \
                ((1 - self.update_rate) * normed_values[:, unique_related_bank_idx] + \
                 self.update_rate * val_bank_update[:, unique_related_bank_idx])

            # less than the threshold, concat them
            selected_idx = (related_bank_corr[0] <= self.thres_close).nonzero()

            if self.class_budget < bank_n + selected_idx.shape[0]:
                self.remove(class_idx, selected_idx.shape[0], frame_idx)

            self.keys[class_idx] = torch.cat([self.keys[class_idx], prev_key[class_idx][:, selected_idx[:, 0]]], dim=1)
            self.values[class_idx] = \
                torch.cat([self.values[class_idx], prev_value[class_idx][:, selected_idx[:, 0]]], dim=1)

            new_info = torch.zeros((selected_idx.shape[0], 2), device=self.device)
            new_info[:, 0] = frame_idx
            self.info[class_idx] = torch.cat([self.info[class_idx], new_info], dim=0)

            self.peak_n[class_idx] = max(self.peak_n[class_idx], self.info[class_idx].shape[0])

            self.info[class_idx][:, 1] = torch.clamp(self.info[class_idx][:, 1], 0, 1e5)  # Prevent inf

    def remove(self, class_idx, request_n, frame_idx):

        old_size = self.keys[class_idx].shape[1]

        LFU = frame_idx - self.info[class_idx][:, 0]  # time length
        LFU = self.info[class_idx][:, 1] / LFU
        thres_dynamic = int(LFU.min()) + 1
        iter_cnt = 0

        while True:
            selected_idx = LFU > thres_dynamic
            self.keys[class_idx] = self.keys[class_idx][:, selected_idx]
            self.values[class_idx] = self.values[class_idx][:, selected_idx]
            self.info[class_idx] = self.info[class_idx][selected_idx]
            LFU = LFU[selected_idx]
            iter_cnt += 1

            balance = (self.class_budget - self.keys[class_idx].shape[1]) - request_n
            if balance < 0:
                thres_dynamic = int(LFU.min()) + 1
            else:
                break

        new_size = self.keys[class_idx].shape[1]
        self.replace_n[class_idx] += old_size - new_size

        return balance

    def print_peak_mem(self):

        ur = self.peak_n / self.class_budget
        rr = self.replace_n / self.class_budget
        print(f'Obj num: {self.obj_n}.', f'Budget / obj: {self.class_budget}.', f'UR: {ur}.', f'Replace: {rr}.')
