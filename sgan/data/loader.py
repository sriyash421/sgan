from torch.utils.data import DataLoader

# from sgan.data.trajectories import TrajectoryDataset
from sgan.data.conditional_trajectories import ConditionalTrajectoryDataset, seq_collate

def data_loader(args, path):
    dset = ConditionalTrajectoryDataset(
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
        collate_fn=seq_collate)
    return dset, loader
