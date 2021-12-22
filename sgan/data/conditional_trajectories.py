import logging
import os
import math

import numpy as np

import torch
from .trajectories import TrajectoryDataset

logger = logging.getLogger(__name__)


def pop_row(T, i):
    # print("before: ", T.shape, " pop: ", i)
    r = T[i].clone()
    T = torch.cat([T[0:i], T[i+1:]])
    # print("after: ", T.shape, " r: ", r.shape, "\n")
    return T, r


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, ego_seq_list, ego_seq_rel_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    ego_traj = torch.stack(ego_seq_list, dim=0).permute(2, 0, 1)
    ego_traj_rel = torch.stack(ego_seq_rel_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, ego_traj, ego_traj_rel
    ]

    return tuple(out)


class ConditionalTrajectoryDataset(TrajectoryDataset):
    """Dataloder for the Conditional Trajectory datasets"""

    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(ConditionalTrajectoryDataset, self).__init__(data_dir, obs_len,
                                                           pred_len, skip, threshold, min_ped, delim)

        # sequences x peds in the sequence
        self.possible_ego_agents = np.concatenate(
            [np.arange(p) for p in self.num_peds_in_seq])
        self.idx_to_sequence = np.concatenate([np.ones(
            p, dtype=np.int)*i for i, p in enumerate(self.num_peds_in_seq)])  # sequences x peds in the sequence

    def __len__(self):
        return self.possible_ego_agents.size

    def __getitem__(self, idx):

        index = self.idx_to_sequence[idx]
        start, end = self.seq_start_end[index]
        ego_id = self.possible_ego_agents[idx]

        obs_traj, ego_traj = pop_row(self.obs_traj[start:end, :], ego_id)
        pred_traj, ego_traj_future = pop_row(
            self.pred_traj[start:end, :], ego_id)
        obs_traj_rel, ego_traj_rel = pop_row(
            self.obs_traj_rel[start:end, :], ego_id)
        pred_traj_rel, ego_traj_rel_future = pop_row(
            self.pred_traj_rel[start:end, :], ego_id)
        non_linear_ped, _ = pop_row(self.non_linear_ped[start:end], ego_id)
        loss_mask, _ = pop_row(self.loss_mask[start:end, :], ego_id)

        ego_traj = torch.cat((ego_traj, ego_traj_future), axis=1)
        ego_traj_rel = torch.cat((ego_traj_rel, ego_traj_rel_future), axis=1)

        out = [
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
            non_linear_ped, loss_mask, ego_traj, ego_traj_rel
        ]
        # print([o.shape for o in out])
        return out
