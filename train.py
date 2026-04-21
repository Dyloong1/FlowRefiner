#!/usr/bin/env python3
"""
FlowRefiner training script.

    python train.py --data_source jhu --data_dir <path> \
        --refiner_steps 2 --sigma_schedule fixed_range \
        --epochs 150 --checkpoint_dir checkpoints/flowrefiner_jhu
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from Model.models import get_model
from Model.configs.flowrefiner_config import get_config
from Dataset.jhu_dataset import SparseJHUDataset


class ExponentialMovingAverage:
    """EMA for model weights (decay=0.995, matches PDE-Refiner reference)."""

    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, overwrite=False):
        if len(self.shadow) > 0 and not overwrite:
            return
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * p.data.detach() \
                                    + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data
                p.data = self.shadow[name]

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[name]
        self.backup = {}


def parse_args():
    p = argparse.ArgumentParser(description='FlowRefiner training')

    p.add_argument('--config', type=str, default='base',
                   choices=['small', 'base', 'large'],
                   help='Model size (base ≈ 50M params)')

    # Data
    p.add_argument('--data_source', type=str, default='jhu', choices=['jhu', 'dns'])
    p.add_argument('--data_dir', type=str, required=True,
                   help='Root directory holding the flow-field .npy files')
    p.add_argument('--split_config', type=str, default=None,
                   help='Path to AR split JSON (defaults to Dataset/jhu_data_splits_ar.json)')

    # Training
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)

    # Checkpoints
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--resume', type=str, default=None,
                   help='Resume training from checkpoint (keeps optimizer/scheduler/epoch)')
    p.add_argument('--finetune', type=str, default=None,
                   help='Load model+EMA weights only; reset optimizer/scheduler/epoch')

    # FlowRefiner-specific overrides
    p.add_argument('--refiner_steps', type=int, default=None,
                   help='K, number of refinement steps (overrides config)')
    p.add_argument('--sigma_schedule', type=str, default=None,
                   choices=['ddpm', 'ddpm_large', 'linear', 'fixed_range'])
    p.add_argument('--ode_steps', type=int, default=None,
                   help='N Euler substeps per refinement k at inference')
    p.add_argument('--sigma_max', type=float, default=None,
                   help='For fixed_range: upper bound (default 0.01)')
    p.add_argument('--sigma_min', type=float, default=None,
                   help='For fixed_range: lower bound (default 0.001)')
    p.add_argument('--noise_type', type=str, default='white',
                   help='Flow matching prior. See utils/noise_generators.py')
    p.add_argument('--input_timesteps', type=int, default=None,
                   help='Override INPUT_TIMESTEPS (default 5)')

    # Logging
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='flowrefiner')

    args = p.parse_args()

    if args.split_config is None:
        if args.data_source == 'jhu':
            args.split_config = str(CODE_DIR / 'Dataset' / 'jhu_data_splits_ar.json')
        else:
            args.split_config = str(CODE_DIR / 'Dataset' / 'dns_data_splits_ar.json')

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


def create_dataloaders(args, norm_stats):
    if args.data_source == 'jhu':
        train_ds = SparseJHUDataset(
            data_dir=args.data_dir, split='train',
            split_config_path=args.split_config,
            indicators=['u', 'v', 'w', 'p'],
            normalize=True, norm_stats=norm_stats, dense_z_size=128,
        )
        val_ds = SparseJHUDataset(
            data_dir=args.data_dir, split='val',
            split_config_path=args.split_config,
            indicators=['u', 'v', 'w', 'p'],
            normalize=True, norm_stats=norm_stats, dense_z_size=128,
        )
    else:
        from Dataset.sparse_dataset import SparseDNSDataset
        train_ds = SparseDNSDataset(
            data_dir=args.data_dir, split='train',
            split_config_path=args.split_config,
            indicators=['u', 'v', 'w', 'p'], normalize=True,
        )
        val_ds = SparseDNSDataset(
            data_dir=args.data_dir, split='val',
            split_config_path=args.split_config,
            indicators=['u', 'v', 'w', 'p'], normalize=True,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device, ema, input_timesteps=None):
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        x = batch['input_dense'].to(device)
        y = batch['output_dense'].to(device)
        if input_timesteps is not None and x.shape[1] > input_timesteps:
            x = x[:, -input_timesteps:]
        optimizer.zero_grad()
        loss = model.get_training_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update()
        total += loss.item()
        n += 1
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, input_timesteps=None):
    model.eval()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc='Validation'):
        x = batch['input_dense'].to(device)
        y = batch['output_dense'].to(device)
        if input_timesteps is not None and x.shape[1] > input_timesteps:
            x = x[:, -input_timesteps:]
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Config & overrides
    config = get_config(args.config)
    if args.data_source == 'dns':
        config.TOTAL_Z_LAYERS = 64
    if args.refiner_steps is not None:
        config.REFINER_STEPS = args.refiner_steps
        print(f'[K] REFINER_STEPS = {args.refiner_steps}')
    if args.sigma_schedule is not None:
        config.SIGMA_SCHEDULE = args.sigma_schedule
        print(f'[sigma] SIGMA_SCHEDULE = {args.sigma_schedule}')
    if args.ode_steps is not None:
        config.ODE_STEPS = args.ode_steps
    if args.sigma_max is not None:
        config.SIGMA_MAX = args.sigma_max
    if args.sigma_min is not None:
        config.SIGMA_MIN = args.sigma_min
    if args.noise_type != 'white':
        config.NOISE_TYPE = args.noise_type
        print(f'[noise] NOISE_TYPE = {args.noise_type}')
    if args.input_timesteps is not None:
        config.INPUT_TIMESTEPS = args.input_timesteps

    print(f'Config: {config}')

    norm_stats = load_norm_stats(args.data_source)
    train_loader, val_loader = create_dataloaders(args, norm_stats)

    model = get_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'FlowRefiner parameters: {n_params:,}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val = float('inf')
    ema_source = None
    if args.resume:
        print(f'Resuming from {args.resume}')
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        sched_state = ck['scheduler_state_dict']
        sched_state['T_max'] = args.epochs
        scheduler.load_state_dict(sched_state)
        start_epoch = ck['epoch'] + 1
        best_val = ck.get('best_val_loss', float('inf'))
        ema_source = args.resume
    elif args.finetune:
        print(f'Finetuning from {args.finetune}')
        ck = torch.load(args.finetune, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        ema_source = args.finetune

    # EMA
    ema = ExponentialMovingAverage(model, decay=0.995)
    ema.register()
    if ema_source is not None:
        ck = torch.load(ema_source, map_location=device)
        if ck.get('ema_shadow') is not None:
            ema.shadow = ck['ema_shadow']
            print('  EMA shadow restored')
        del ck

    # Training log
    log_path = ckpt_dir / 'training_log.csv'
    log_file = open(log_path, 'a' if args.resume else 'w', newline='')
    log_writer = csv.writer(log_file)
    if not args.resume:
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val', 'lr'])

    # Save config snapshot
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump({'args': vars(args),
                   'config': {k: getattr(config, k) for k in vars(config)},
                   'n_params': n_params}, f, indent=2, default=str)

    # wandb (lightweight)
    wandb_run = None
    if args.use_wandb:
        import os
        import wandb
        os.environ['WANDB_DIR'] = str(ckpt_dir)
        run_name = ckpt_dir.name
        wandb_run = wandb.init(
            project=args.wandb_project, name=run_name, config=vars(args),
            resume='allow', id=run_name.replace('/', '_'),
            settings=wandb.Settings(_save_requirements=False, console='off'),
            save_code=False,
        )

    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss = train_one_epoch(model, train_loader, optimizer, device, ema,
                                     input_timesteps=args.input_timesteps)
        if ema is not None:
            ema.apply_shadow()
        val_loss = validate(model, val_loader, device,
                            input_timesteps=args.input_timesteps)
        if ema is not None:
            ema.restore()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        print(f'  train={train_loss:.6f}  val={val_loss:.6f}  '
              f'best={best_val:.6f}  lr={lr:.2e}'
              f'{"  [BEST]" if is_best else ""}')
        log_writer.writerow([epoch + 1, train_loss, val_loss, best_val, lr])
        log_file.flush()

        if wandb_run is not None:
            import wandb
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss,
                       'best_val': best_val, 'lr': lr}, step=epoch + 1)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss, 'val_loss': val_loss,
            'best_val_loss': best_val,
            'config': {k: getattr(config, k) for k in vars(config)},
            'model_name': 'flowrefiner',
            'ema_shadow': ema.shadow if ema is not None else None,
        }
        torch.save(checkpoint, ckpt_dir / 'latest.pt')
        if is_best:
            if ema is not None:
                ema.apply_shadow()
                best_ckpt = dict(checkpoint)
                best_ckpt['model_state_dict'] = model.state_dict()
                ema.restore()
                torch.save(best_ckpt, ckpt_dir / 'best.pt')
            else:
                torch.save(checkpoint, ckpt_dir / 'best.pt')

    log_file.close()
    if wandb_run is not None:
        import wandb
        wandb.finish()
    print(f'\nTraining complete. Best val loss = {best_val:.6f}')
    print(f'Checkpoints saved to {ckpt_dir}')


if __name__ == '__main__':
    main()
