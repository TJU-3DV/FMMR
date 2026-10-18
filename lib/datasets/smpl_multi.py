"""EgoHumans multi-person dataset used by embedding training scripts."""
import os
import random

import numpy as np
import torch

from vggt.utils.load_fn import load_and_preprocess_images


def toDevice(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    if sample is None or type(sample) == str:
        return sample
    if isinstance(sample, dict):
        return {k: toDevice(v, device) for k, v in sample.items()}
    return [toDevice(s, device) for s in sample]


class SMPLDataset_multi(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        base = "/mnt/data2/yyw/data_multiview/yyw/egohuman/yyw/"
        big_squence_list = ["02_lego", "03_fencing"]
        squence_list = []
        big_index = []
        num_list = []
        cam_list = []
        for idx in range(len(big_squence_list)):
            ll = sorted(os.listdir(f"{base}/{big_squence_list[idx]}"))[:1]
            squence_list.extend(ll)
            big_index.extend([idx for i in range(len(ll))])
        for i in range(len(squence_list)):
            num_list.append(len(os.listdir(
                f"{base}/{big_squence_list[big_index[i]]}/{squence_list[i]}/smpl"
            )))
            cam_list.append(np.load(
                f"{base}/{big_squence_list[big_index[i]]}/{squence_list[i]}/cameras.npy",
                allow_pickle=True,
            ).item())
        for i in range(1, len(num_list)):
            num_list[i] = num_list[i] + num_list[i - 1]
        self.big_squence_list = big_squence_list
        self.squence_list = squence_list
        self.big_index = big_index
        self.num_list = num_list
        self.cam_list = cam_list
        self.base = base
        self.faces = torch.from_numpy(np.load(
            "/mnt/data2/yyw/data_multiview/yyw/repo/egohumans/egohumans/assets/smpl_faces.npy"
        ).astype(np.int32))

    def __len__(self):
        return self.num_list[-1]

    def __getitem__(self, index):
        for idx, num in enumerate(self.num_list):
            if index < num:
                break
        if idx == 0:
            frame = index + 1
        else:
            frame = index - self.num_list[idx - 1] + 1
        image_names = []
        gt_extri = []
        gt_intri = []
        camlist = random.sample(list(self.cam_list[idx].keys()), 6)
        for cc in camlist:
            image_names.append(
                f"{self.base}/{self.big_squence_list[self.big_index[idx]]}/"
                f"{self.squence_list[idx]}/rgb/{cc}/{frame:05d}.jpg"
            )
            gt_extri.append(self.cam_list[idx][cc]["extrinsics"])
            gt_intri.append(self.cam_list[idx][cc]["intrinsics"])
        gt_extri = torch.from_numpy(np.array(gt_extri)).float()
        gt_intri = torch.from_numpy(np.array(gt_intri)).float()
        images = load_and_preprocess_images(image_names)
        smpl = np.load(
            f"{self.base}/{self.big_squence_list[self.big_index[idx]]}/"
            f"{self.squence_list[idx]}/smpl/{frame:05d}.npy",
            allow_pickle=True,
        ).item()
        res = {
            "img": images,
            "smpl_verts": [torch.from_numpy(smpl[key]["vertices"]) for key in smpl.keys()],
            "c2w": gt_extri.inverse(),
            "K": gt_intri,
        }
        return res
