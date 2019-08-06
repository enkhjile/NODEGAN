import lmdb
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from . import get_paths_from_lmdb, read_img_lmdb, augment, modcrop


class TrainDataset(Dataset):
    def __init__(self, GT_dir, LQ_dir, crop_size=128, scale=4):
        super(TrainDataset, self).__init__()
        self.paths_GT, self.sizes_GT = get_paths_from_lmdb(GT_dir)
        self.paths_LQ, self.sizes_LQ = get_paths_from_lmdb(LQ_dir)
        self.GT_env = lmdb.open(GT_dir, readonly=True, lock=False,
                                readahead=False, meminit=False)
        self.LQ_env = lmdb.open(LQ_dir, readonly=True, lock=False,
                                readahead=False, meminit=False)
        self.crop_size = crop_size
        self.scale = scale

    def __getitem__(self, index):
        GT_path, LQ_path = self.paths_GT[index], self.paths_LQ[index]
        GT_size, LQ_size = self.sizes_GT[index], self.sizes_LQ[index]

        GT_size = [int(s) for s in GT_size.split('_')]
        LQ_size = [int(s) for s in LQ_size.split('_')]

        img_GT = read_img_lmdb(self.GT_env, GT_path, GT_size)
        img_LQ = read_img_lmdb(self.LQ_env, LQ_path, LQ_size)

        H, W, C = img_LQ.shape
        LQ_size = self.crop_size // self.scale

        # randomly crop
        rnd_h = random.randint(0, max(0, H - LQ_size))
        rnd_w = random.randint(0, max(0, W - LQ_size))
        img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        rnd_h_GT = int(rnd_h * self.scale)
        rnd_w_GT = int(rnd_w * self.scale)
        img_GT = img_GT[rnd_h_GT:rnd_h_GT + self.crop_size,
                        rnd_w_GT:rnd_w_GT + self.crop_size, :]

        # augmentation - flip, rotate
        img_LQ, img_GT = augment([img_LQ, img_GT])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]] / 255
        img_LQ = img_LQ[:, :, [2, 1, 0]] / 255
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        return {'img_GT': img_GT, 'img_LQ': img_LQ}

    def __len__(self):
        return len(self.paths_GT)


class ValDataset(Dataset):
    def __init__(self, GT_dir, LQ_dir, LQ_dir_r, scale=4):
        super(ValDataset, self).__init__()
        self.paths_GT, self.sizes_GT = get_paths_from_lmdb(GT_dir)
        self.paths_LQ, self.sizes_LQ = get_paths_from_lmdb(LQ_dir)
        self.paths_LQ_r, self.sizes_LQ_r = get_paths_from_lmdb(LQ_dir_r)
        self.GT_env = lmdb.open(GT_dir, readonly=True, lock=False,
                                readahead=False, meminit=False)
        self.LQ_env = lmdb.open(LQ_dir, readonly=True, lock=False,
                                readahead=False, meminit=False)
        self.LQ_env_r = lmdb.open(LQ_dir_r, readonly=True, lock=False,
                                  readahead=False, meminit=False)
        self.scale = scale

    def __getitem__(self, index):
        GT_path, LQ_path, LQ_path_r = self.paths_GT[index], \
            self.paths_LQ[index], self.paths_LQ_r[index]
        GT_size, LQ_size, LQ_size_r = self.sizes_GT[index], \
            self.sizes_LQ[index], self.sizes_LQ_r[index]

        GT_size = [int(s) for s in GT_size.split('_')]
        LQ_size = [int(s) for s in LQ_size.split('_')]
        LQ_size_r = [int(s) for s in LQ_size_r.split('_')]

        img_GT = read_img_lmdb(self.GT_env, GT_path, GT_size)
        img_LQ = read_img_lmdb(self.LQ_env, LQ_path, LQ_size)
        img_LQ_r = read_img_lmdb(self.LQ_env_r, LQ_path_r, LQ_size_r)

        img_GT = modcrop(img_GT, self.scale)
        img_LQ_r = modcrop(img_LQ_r, self.scale)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]] / 255
        img_LQ = img_LQ[:, :, [2, 1, 0]] / 255
        img_LQ_r = img_LQ_r[:, :, [2, 1, 0]] / 255
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_LQ_r = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ_r, (2, 0, 1)))).float()

        return {'img_GT': img_GT, 'img_LQ': img_LQ, 'img_LQ_r': img_LQ_r}

    def __len__(self):
        return len(self.paths_GT)
