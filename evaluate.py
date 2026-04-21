#!/usr/bin/env python3
"""
FlowRefiner autoregressive (AR) evaluation.

Reproduces the per-round rollout protocol from the paper:
    Round 1: GT input [t0-t4]  -> predict [t5-t9]
    Round 2: predicted [t5-t9] -> predict [t10-t14]
    Round 3: predicted [t10-t14] -> predict [t15-t19]  (JHU only)

DNS (TGV) supports up to 2 AR rounds (15 continuous test frames);
JHU supports up to 3 AR rounds (20 continuous test frames).

Usage:
    python evaluate.py --data_source jhu --data_dir <path> \
        --checkpoint_dir checkpoints/flowrefiner_jhu --ckpt_type best \
        --refiner_steps 2 --sigma_schedule fixed_range
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from Model.models import get_model
from Model.configs.flowrefiner_config import get_config
from utils.metrics import compute_ssim_3d
from utils.physics_utils import divergence_stats

DNS_TRUNC_Z = 64


def parse_args():
    p = argparse.ArgumentParser(description='FlowRefiner AR evaluation')
    p.add_argument('--config', type=str, default='base',
                   choices=['small', 'base', 'large'])
    p.add_argument('--data_source', type=str, default='jhu', choices=['jhu', 'dns'])
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--split_config', type=str, default=None)
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--ckpt_type', type=str, default='best',
                   choices=['best', 'latest'])
    p.add_argument('--max_ar_rounds', type=int, default=None)
    p.add_argument('--out_dir', type=str, default='results')

    # FlowRefiner overrides (must match training to rebuild scheduler correctly)
    p.add_argument('--refiner_steps', type=int, default=None)
    p.add_argument('--sigma_schedule', type=str, default=None,
                   choices=['ddpm', 'ddpm_large', 'linear', 'fixed_range'])
    p.add_argument('--ode_steps', type=int, default=None)
    p.add_argument('--sigma_max', type=float, default=None)
    p.add_argument('--sigma_min', type=float, default=None)
    p.add_argument('--noise_type', type=str, default='white')
    p.add_argument('--input_timesteps', type=int, default=None)
    args = p.parse_args()
    if args.split_config is None:
        args.split_config = str(
            CODE_DIR / 'Dataset' / ('jhu_data_splits_ar.json'
                                    if args.data_source == 'jhu'
                                    else 'dns_data_splits_ar.json'))
    return args


def load_norm_stats(data_source):
    path = CODE_DIR / 'Dataset' / (
        'jhu_normalization_stats.json' if data_source == 'jhu'
        else 'dns_normalization_stats.json'
    )
    if not path.exists():
        print(f'WARNING: normalization stats not found at {path}; using identity.')
        return None
    with open(path) as f:
        raw = json.load(f)
    inds = raw.get('indicators', raw.get('channels', ['u', 'v', 'w', 'p']))
    return {ind: {'mean': raw['mean'][i], 'std': raw['std'][i]}
            for i, ind in enumerate(inds)}


def load_frame(data_source, data_dir, indicator, fidx, norm_stats):
    if data_source == 'dns':
        fp = Path(data_dir) / f'img_{indicator}_dns{fidx}.npy'
        data = np.load(fp).astype(np.float32)[:DNS_TRUNC_Z, :, :]
    else:
        ts = round(5.02 + fidx * 0.01, 3)
        fp = Path(data_dir) / f'{indicator}_{ts:.3f}.npy'
        data = np.load(fp).astype(np.float32)
    if norm_stats and indicator in norm_stats:
        data = (data - norm_stats[indicator]['mean']) / (norm_stats[indicator]['std'] + 1e-8)
    return data


def load_sequence(data_source, data_dir, frame_indices, indicators, norm_stats):
    all_ch = []
    for ind in indicators:
        frames = [load_frame(data_source, data_dir, ind, fi, norm_stats)
                  for fi in frame_indices]
        all_ch.append(np.stack(frames, axis=0))
    seq = np.stack(all_ch, axis=1)  # (T, C, Z, H, W)
    return torch.from_numpy(seq)


def denormalize(tensor, indicators, norm_stats):
    if norm_stats is None:
        return tensor
    out = tensor.clone()
    for c, ind in enumerate(indicators):
        if ind in norm_stats:
            out[:, c] = out[:, c] * norm_stats[ind]['std'] + norm_stats[ind]['mean']
    return out


def compute_metrics(pred, gt, indicators, norm_stats):
    """pred, gt : (T, C, Z, H, W) normalized."""
    pred_phys = denormalize(pred, indicators, norm_stats)
    gt_phys = denormalize(gt, indicators, norm_stats)

    mse = ((pred_phys - gt_phys) ** 2).mean().item()
    mae = (pred_phys - gt_phys).abs().mean().item()
    rmse = float(np.sqrt(mse))
    rel_l2 = (torch.norm(pred_phys - gt_phys)
              / (torch.norm(gt_phys) + 1e-8)).item()

    per_channel = {}
    ssim_vals = []
    for c, ind in enumerate(indicators):
        p = pred_phys[:, c]
        g = gt_phys[:, c]
        ch_mse = ((p - g) ** 2).mean().item()
        ch_rel = (torch.norm(p - g) / (torch.norm(g) + 1e-8)).item()
        ch_ssim = compute_ssim_3d(p.unsqueeze(1), g.unsqueeze(1))
        ssim_vals.append(ch_ssim)
        per_channel[ind] = {
            'mse': ch_mse, 'mae': (p - g).abs().mean().item(),
            'rmse': float(np.sqrt(ch_mse)),
            'rel_l2': ch_rel, 'ssim': ch_ssim,
        }

    physics = {}
    vel_idx = [i for i, ind in enumerate(indicators) if ind in ('u', 'v', 'w')]
    if len(vel_idx) == 3:
        pred_vel = pred_phys[:, vel_idx]
        gt_vel = gt_phys[:, vel_idx]
        dmax, dmean = divergence_stats(pred_vel)
        physics['div_max'] = float(dmax.mean().item())
        physics['div_mean'] = float(dmean.mean().item())
        E_p = 0.5 * (pred_vel ** 2).sum(dim=1).mean().item()
        E_g = 0.5 * (gt_vel ** 2).sum(dim=1).mean().item()
        physics['dE_pct'] = float((E_p - E_g) / (E_g + 1e-8) * 100)
        physics['dMeanVel'] = float((pred_vel.mean() - gt_vel.mean()).abs().item())

    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'rel_l2': rel_l2,
            'ssim': float(np.mean(ssim_vals)),
            'per_channel': per_channel, 'physics': physics}


@torch.no_grad()
def run_ar_eval(model, data_source, data_dir, split_config,
                norm_stats, indicators, device, max_ar_rounds=None,
                input_timesteps=None):
    model.eval()
    with open(split_config) as f:
        cfg = json.load(f)
    test_sequences = cfg['test_sequences']

    round_metrics = {}
    for seq in test_sequences:
        chunk_id = seq['chunk_id']
        f0, f1 = seq['frame_range']
        ar_rounds = seq['supports_ar_rounds']
        if max_ar_rounds is not None:
            ar_rounds = min(ar_rounds, max_ar_rounds)
        total = 5 + ar_rounds * 5
        indices = list(range(f0, f0 + total))
        print(f'\n  Chunk {chunk_id}: frames {f0}-{f0 + total - 1}, '
              f'AR rounds: {ar_rounds}')
        full_seq = load_sequence(data_source, data_dir, indices,
                                 indicators, norm_stats)
        current = full_seq[:5].unsqueeze(0).to(device)
        if input_timesteps is not None and current.shape[1] > input_timesteps:
            current = current[:, -input_timesteps:]
        for r in range(1, ar_rounds + 1):
            pred = model(current)
            gt = full_seq[r * 5:(r + 1) * 5]
            m = compute_metrics(pred[0].cpu(), gt, indicators, norm_stats)
            round_metrics.setdefault(r, []).append(m)
            phys_s = ''
            if m.get('physics'):
                phys = m['physics']
                phys_s = f", div_max={phys.get('div_max',0):.3f}, dE={phys.get('dE_pct',0):.2f}%"
            print(f'    Round {r}: RMSE={m["rmse"]:.5f}, '
                  f'SSIM={m["ssim"]:.4f}, RelL2={m["rel_l2"]:.5f}{phys_s}')
            current = pred
            if input_timesteps is not None and current.shape[1] > input_timesteps:
                current = current[:, -input_timesteps:]

    results = {}
    for r, mlist in sorted(round_metrics.items()):
        agg = {
            'mse': float(np.mean([m['mse'] for m in mlist])),
            'mae': float(np.mean([m['mae'] for m in mlist])),
            'rmse': float(np.sqrt(np.mean([m['mse'] for m in mlist]))),
            'rel_l2': float(np.mean([m['rel_l2'] for m in mlist])),
            'ssim': float(np.mean([m['ssim'] for m in mlist])),
            'num_chunks': len(mlist),
            'per_channel': {},
        }
        for ind in indicators:
            agg['per_channel'][ind] = {
                'mse': float(np.mean([m['per_channel'][ind]['mse'] for m in mlist])),
                'mae': float(np.mean([m['per_channel'][ind]['mae'] for m in mlist])),
                'rmse': float(np.sqrt(np.mean([m['per_channel'][ind]['mse'] for m in mlist]))),
                'rel_l2': float(np.mean([m['per_channel'][ind]['rel_l2'] for m in mlist])),
                'ssim': float(np.mean([m['per_channel'][ind]['ssim'] for m in mlist])),
            }
        if mlist[0].get('physics'):
            agg['physics'] = {k: float(np.mean([m['physics'][k] for m in mlist]))
                              for k in mlist[0]['physics']}
        results[f'round_{r}'] = agg
    return results


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    config = get_config(args.config)
    if args.data_source == 'dns':
        config.TOTAL_Z_LAYERS = 64
    if args.refiner_steps is not None:
        config.REFINER_STEPS = args.refiner_steps
    if args.sigma_schedule is not None:
        config.SIGMA_SCHEDULE = args.sigma_schedule
    if args.ode_steps is not None:
        config.ODE_STEPS = args.ode_steps
    if args.sigma_max is not None:
        config.SIGMA_MAX = args.sigma_max
    if args.sigma_min is not None:
        config.SIGMA_MIN = args.sigma_min
    if args.noise_type != 'white':
        config.NOISE_TYPE = args.noise_type
    if args.input_timesteps is not None:
        config.INPUT_TIMESTEPS = args.input_timesteps

    ckpt_path = Path(args.checkpoint_dir) / f'{args.ckpt_type}.pt'
    if not ckpt_path.exists():
        print(f'Checkpoint not found: {ckpt_path}')
        sys.exit(1)

    norm_stats = load_norm_stats(args.data_source)
    indicators = ['u', 'v', 'w', 'p']

    model = get_model(config)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    print(f'Loaded {ckpt_path} (epoch {ck.get("epoch", "?")})')

    max_rounds = args.max_ar_rounds or (2 if args.data_source == 'dns' else 3)

    results = run_ar_eval(
        model, args.data_source, args.data_dir, args.split_config,
        norm_stats, indicators, device, max_ar_rounds=max_rounds,
        input_timesteps=args.input_timesteps,
    )

    print(f'\n{"="*60}')
    print('AR Evaluation Summary')
    print(f'{"="*60}')
    for rn, m in sorted(results.items()):
        print(f'\n  {rn} ({m["num_chunks"]} chunks)')
        print(f'    RMSE={m["rmse"]:.5f}  SSIM={m["ssim"]:.4f}  RelL2={m["rel_l2"]:.5f}')
        if m.get('physics'):
            ph = m['physics']
            print(f'    div_max={ph.get("div_max",0):.3f}  div_mean={ph.get("div_mean",0):.3f}  '
                  f'dE={ph.get("dE_pct",0):+.2f}%')
        for ind in indicators:
            ch = m['per_channel'][ind]
            print(f'      {ind}: RMSE={ch["rmse"]:.5f}  SSIM={ch["ssim"]:.4f}  RelL2={ch["rel_l2"]:.5f}')

    out_dir = Path(args.out_dir) / Path(args.checkpoint_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'eval_ar_{args.ckpt_type}.json'
    save_data = {
        'config': args.config,
        'data_source': args.data_source,
        'ckpt_type': args.ckpt_type,
        'max_ar_rounds': max_rounds,
        'epoch': ck.get('epoch'),
        'refiner_steps': config.REFINER_STEPS,
        'sigma_schedule': config.SIGMA_SCHEDULE,
        'ode_steps': config.ODE_STEPS,
        'noise_type': config.NOISE_TYPE,
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
