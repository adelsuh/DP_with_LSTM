import os
import glob
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from equibot.policies.utils.misc import rotate_around_z
from equibot.policies.datasets.dataset import BaseDataset


class LSTMDataset(BaseDataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__(cfg, mode)
        key_fn = lambda x: "_".join(x.split("/")[-1].split("_")[:-1])
        self.ep_list = list(sorted(set([key_fn(fn) for fn in self.file_names])))
        self.path = cfg["path"]

    def __getitem__(self, idx): #Get full trajectory
        ep = self.ep_list[idx]
        ep_t_list = np.arange(0, self.ep_length_dict[ep] - 1)
        ret = dict(pc=[], rgb=[], eef_pos=[], eef_rot=[], action=[], offset=[])
        if self.num_augment > 0:
            if self.same_aug_per_sample:
                aug_idx = np.random.randint(self.num_augment)
            else:
                aug_idx = idx * self.num_augment + np.random.randint(self.num_augment)
        else:
            aug_idx = None
        for t in ep_t_list:
            if self.use_four_digit_time:
                fn_t = "_".join(
                    [os.path.join(self.path, ep)]+ [f"t{t:04d}.npz"]
                )
            else:
                fn_t = "_".join(
                    [os.path.join(self.path, ep)] + [f"t{t:02d}.npz"]
                )

            keys = ["action", "eef_pos", "pc"]
            data_t = self._process_data_from_file(fn_t, keys, aug_idx=aug_idx)

            for k in data_t.keys():
                if k in ret:
                    ret[k].append(data_t[k])
        ret = {k: np.array(v) for k, v in ret.items() if len(v) > 0}

        # assert len(ret["pc"]) == self.obs_horizon
        # assert len(ret["rgb"]) == self.obs_horizon
        # assert len(ret["eef_pos"]) == self.obs_horizon
        # assert len(ret["action"]) == self.pred_horizon

        return ret
    
    def __len__(self):
        return len(self.ep_list)

    # def __getitem__(self, idx):
    #     fn = self.file_names[idx]
    #     key_fn = lambda x: "_".join(x.split("/")[-1].split("_")[:-1])
    #     offset_t = self.ep_t_offset_dict[key_fn(fn)]
    #     ep_t = int(fn.split("_")[-1][1:-4]) - offset_t
    #     start_t = 0
    #     end_t = ep_t - (self.obs_horizon - 1)
    #     ep_t_list = np.arange(start_t, end_t)
    #     clipped_ep_t_list = np.clip(ep_t_list, 0, self.ep_length_dict[key_fn(fn)] - 1)
    #     ret_past = dict(pc_past=[], eef_past=[], offset=[], past_length=0)
    #     if self.num_augment > 0:
    #         if self.same_aug_per_sample:
    #             aug_idx = np.random.randint(self.num_augment)
    #         else:
    #             aug_idx = idx * self.num_augment + np.random.randint(self.num_augment)
    #     else:
    #         aug_idx = None
    #     for t, clipped_t in zip(ep_t_list, clipped_ep_t_list):
    #         if self.use_four_digit_time:
    #             fn_t = "_".join(
    #                 fn.split("_")[:-1] + [f"t{clipped_t + offset_t:04d}.npz"]
    #             )
    #         else:
    #             fn_t = "_".join(
    #                 fn.split("_")[:-1] + [f"t{clipped_t + offset_t:02d}.npz"]
    #             )
    #         if t == start_t and self.obs_horizon == 2 and self.state_latency > 0:
    #             # take into account observation latency for previous obs
    #             # current timestep is [start_t]
    #             # sample state from range [t + 1 - state_latency, ep_t]

    #             if self.state_latency_random:
    #                 state_t = t - np.random.randint(self.state_latency)
    #             else:
    #                 state_t = t - (self.state_latency - 1)

    #             clipped_state_t = max(0, state_t)

    #             if self.use_four_digit_time:
    #                 fn_state_t = "_".join(
    #                     fn.split("_")[:-1] + [f"t{clipped_state_t + offset_t:04d}.npz"]
    #                 )
    #             else:
    #                 fn_state_t = "_".join(
    #                     fn.split("_")[:-1] + [f"t{clipped_state_t + offset_t:02d}.npz"]
    #                 )

    #             data_t = self._process_data_from_file(
    #                 fn_state_t, ["pc", "eef_pos"], aug_idx=aug_idx
    #             )
    #         else:
    #             keys = ["eef_pos", "pc"]
    #             data_t = self._process_data_from_file(fn_t, keys, aug_idx=aug_idx)
            
    #         ret_past["eef_past"].append(data_t["eef_pos"])
    #         ret_past["pc_past"].append(data_t["pc"])
        
    #     ret_past["past_length"] = len(ret_past["eef_past"])

    #     ret_past = {k: np.array(v) if type(v) is list else v for k, v in ret_past.items()}
    #     ret = super().__getitem__(idx)

    #     ret.update(ret_past)

    #     return ret