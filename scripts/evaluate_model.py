import argparse
import os
import torch
import logging
import sys
from attrdict import AttrDict
import time
from sgan.data.loader import data_loader
# from sgan.models import TrajectoryGenerator
from sgan.conditional_models import ConditionalTrajectoryGenerator, OptimizedCSGAN, OptimizedSGAN
from sgan.conditional_models2 import OptimizedCPool
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, optimized_relative_to_abs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=10, type=int)
parser.add_argument('--pred_len', default=3, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--use_gpu', action="store_true")

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def _get_generator(model, checkpoint, device):
    args = AttrDict(checkpoint['args'])
    generator = model(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        device=device)
    generator.load_state_dict(checkpoint['g_state'])
    generator = generator.to(device)
    generator.train()
    # logger.info(generator)
    logger.info(f"{args.pooling_type}")
    return generator

def get_generator(checkpoint, device):
    # try:
    #     return _get_generator(OptimizedCPool, checkpoint, device), True
    # except:
    try:
        return _get_generator(OptimizedCSGAN, checkpoint, device), True
    except:
        return _get_generator(OptimizedSGAN, checkpoint, device), False

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples, device, conditional):
    ade_outer, fde_outer = [], []
    total_traj = 0
    t = []
    with torch.no_grad():
        for batch in loader:
            et = time.time()
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end, ego_traj, ego_traj_rel) = batch
            if conditional:
                pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, seq_start_end, ego_traj, ego_traj_rel, num_samples)
                pred_traj_fake = optimized_relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                
            else:
                pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, seq_start_end, num_samples)
                pred_traj_fake = optimized_relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            et = time.time()-et
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            ade = [displacement_error(pred_traj_fake[i], pred_traj_gt, mode='raw') for i in range(num_samples)]
            fde = [final_displacement_error(pred_traj_fake[i][-1], pred_traj_gt[-1], mode='raw') for i in range(num_samples)]

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

            t.append(et)
            break
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        print(f"Total: {sum(t)} Average per episode: {sum(t)/len(t)}")
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    device = torch.device("cuda" if args.use_gpu else "cpu")
    for path in paths:
        # print(path)
        checkpoint = torch.load(path, map_location=device)
        checkpoint['args']['pred_len'] = args.pred_len
        checkpoint['args']['obs_len'] = args.obs_len
        checkpoint['args']['batch_size'] = 64
        generator, conditional = get_generator(checkpoint, device)
        _args = AttrDict(checkpoint['args'])
        dpath = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, dpath, True)
        ade, fde = evaluate(_args, loader, generator, args.num_samples, device, conditional)
        print('Path: {} Cond: {} Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format( path, conditional,
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
