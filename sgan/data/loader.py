from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate
from sgan.data.conditional_trajectories import ConditionalTrajectoryDataset, cond_seq_collate

def data_loader(args, path, conditional):
    dset = ConditionalTrajectoryDataset if conditional else TrajectoryDataset
    cfunc = cond_seq_collate if conditional else seq_collate
    dset = dset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=cfunc)
    return dset, loader
