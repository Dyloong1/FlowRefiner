"""
稀疏DNS数据集

NOTE: DNS原始数据为65层z，但65不是2的幂次方：
  - cuFFT半精度不支持非2幂次维度 (FNO无法用混合精度)
  - Swin的patch_size和window_size需要能整除z维度
因此统一截断为64层 (丢弃最后一层z=64)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


# DNS数据z维度配置
DNS_RAW_Z_SIZE = 65      # 原始数据z层数
DNS_TRUNCATED_Z_SIZE = 64  # 截断后z层数 (2^6, 便于FFT和patch划分)


class SparseDNSDataset(Dataset):
    """
    DNS数据集

    自动将原始65层z截断为64层
    纯预测任务: input (T_in timesteps) -> output (T_out timesteps)
    """

    def __init__(self,
                 data_dir,
                 split='train',
                 split_config_path=None,
                 indicators=['u', 'v'],
                 normalize=True,
                 norm_stats=None):
        """
        Args:
            data_dir: DNS数据目录
            split: 'train', 'val', 'test'
            split_config_path: 划分配置JSON路径
            indicators: 使用的物理量
            normalize: 是否归一化
            norm_stats: 归一化统计量字典
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.indicators = indicators
        self.normalize = normalize

        if split_config_path is None:
            raise ValueError("必须提供 split_config_path")

        with open(split_config_path, 'r') as f:
            config = json.load(f)

        self.sample_indices = config['splits'][split]
        self.metadata = config['metadata']
        self.input_length = self.metadata['input_length']
        self.output_length = self.metadata['output_length']

        # 使用截断后的z维度 (65→64)
        self.dense_z_size = DNS_TRUNCATED_Z_SIZE

        print(f"\n{split.upper()} 数据集初始化:")
        print(f"  样本数: {len(self.sample_indices)}")
        print(f"  物理量: {indicators}")
        print(f"  z维度: {DNS_RAW_Z_SIZE} → {self.dense_z_size} (截断最后一层)")

        if normalize:
            if norm_stats is None:
                self.norm_stats = {
                    'u': {'mean': 0.0, 'std': 0.249},
                    'v': {'mean': 0.0, 'std': 0.249},
                    'w': {'mean': 0.0, 'std': 0.110},
                    'p': {'mean': -0.046, 'std': 0.073}
                }
            else:
                self.norm_stats = norm_stats
            
            print(f"  归一化: 启用")
            for ind in indicators:
                if ind in self.norm_stats:
                    print(f"    {ind}: mean={self.norm_stats[ind]['mean']:.3f}, "
                          f"std={self.norm_stats[ind]['std']:.3f}")
        else:
            print(f"  归一化: 禁用")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def _load_frame(self, indicator, timestamp):
        """加载单帧数据，并截断z维度65→64"""
        filename = f"img_{indicator}_dns{timestamp}.npy"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        data = np.load(filepath).astype(np.float32)  # (65, 128, 128)

        # 截断z维度: 65 → 64 (丢弃最后一层)
        data = data[:DNS_TRUNCATED_Z_SIZE, :, :]  # (64, 128, 128)

        if self.normalize and indicator in self.norm_stats:
            mean = self.norm_stats[indicator]['mean']
            std = self.norm_stats[indicator]['std']
            data = (data - mean) / (std + 1e-8)

        return data
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'input_dense': (input_length, C, 64, 128, 128) - 已截断
                'output_dense': (output_length, C, 64, 128, 128) - 已截断
                'sample_start': 样本起始帧索引
                'global_timesteps': (input_length + output_length,) 全局时间步序列
            }
        """
        sample = self.sample_indices[idx]
        
        # Handle both formats:
        # 1. Integer start index: sample = 5 -> frames [5,6,7,8,9,10,11,12,13,14]
        # 2. List of frame indices: sample = [0,1,2,3,4,5,6,7,8,9]
        if isinstance(sample, list):
            # sample contains all frame indices for this sample
            all_frames = sample
            input_frames = all_frames[:self.input_length]
            output_frames = all_frames[self.input_length:self.input_length + self.output_length]
            start_idx = all_frames[0]
        else:
            # sample is a start index
            start_idx = sample
            input_frames = list(range(start_idx, start_idx + self.input_length))
            output_frames = list(range(start_idx + self.input_length, 
                                       start_idx + self.input_length + self.output_length))
        
        input_data = []
        for indicator in self.indicators:
            indicator_frames = []
            for t in input_frames:
                frame = self._load_frame(indicator, t)
                indicator_frames.append(frame)
            input_data.append(np.stack(indicator_frames, axis=0))
        
        input_dense = np.stack(input_data, axis=1)
        
        output_data = []
        for indicator in self.indicators:
            indicator_frames = []
            for t in output_frames:
                frame = self._load_frame(indicator, t)
                indicator_frames.append(frame)
            output_data.append(np.stack(indicator_frames, axis=0))
        
        output_dense = np.stack(output_data, axis=1)

        return {
            'input_dense': torch.from_numpy(input_dense),
            'output_dense': torch.from_numpy(output_dense),
            'sample_start': start_idx,
            'global_timesteps': torch.tensor(input_frames + output_frames, dtype=torch.long)
        }


def create_dataloaders(data_dir,
                      split_config_path,
                      indicators=['u', 'v'],
                      batch_size=4,
                      num_workers=4,
                      normalize=True,
                      norm_stats=None):
    """创建训练/验证/测试数据加载器"""
    train_dataset = SparseDNSDataset(
        data_dir=data_dir,
        split='train',
        split_config_path=split_config_path,
        indicators=indicators,
        normalize=normalize,
        norm_stats=norm_stats
    )

    val_dataset = SparseDNSDataset(
        data_dir=data_dir,
        split='val',
        split_config_path=split_config_path,
        indicators=indicators,
        normalize=normalize,
        norm_stats=norm_stats
    )

    test_dataset = SparseDNSDataset(
        data_dir=data_dir,
        split='test',
        split_config_path=split_config_path,
        indicators=indicators,
        normalize=normalize,
        norm_stats=norm_stats
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader