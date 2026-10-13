"""
Authors: Pablo Fernández López, Jorge Martín Pérez
https://github.com/giros-dit/isac-sensing-presence-ml/blob/main/grid_tensor_dataset.py
"""

import argparse
import os
import glob
import pandas as pd
import torch
from torch.fft import fft
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.io import loadmat




coord_classes = ['1_1',
'1_4',
'1_5',
'1_6',
'1_7',
'1_8',
'2_1',
'2_2',
'2_3',
'2_4',
'2_5',
'2_6',
'2_7',
'3_1',
'3_2',
'3_6',
'3_7',
'3_8',
'4_1',
'4_2',
'4_7',
'4_8',
'5_1',
'5_2',
'5_3',
'5_4',
'5_5',
'5_6',
'5_7',
'5_8',
'5_9',
'6_1',
'6_2',
'6_3',
'6_4',
'6_5',
'6_6',
'6_7',
'6_8',
'6_9',
'7_1',
'7_2',
'7_3',
'7_4',
'7_5',
'7_6',
'7_7',
'7_8',
'7_9',
'8_4',
'8_5',
'8_6',
'8_7',
'8_8',
'8_9']










class FAWNGridSymbolTensorDataset(Dataset):
    """Custom Dataset for loading and symbols from grids as tensors with external label CSV"""
    
    def __init__(self, root_dir, data_type=torch.float32):
        """
        Args:
            root_dir: Path to the directory containing all tensor files
            data_type: Data type for tensors (e.g., float32, int64)
        """
        self.root_dir = root_dir
        self.data_type = data_type
        self.classes = coord_classes

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all tensor files from root directory
        self.tensor_files = sorted(glob.glob(os.path.join(root_dir, '*.mat')))
        if len(self.tensor_files) == 0:
            raise RuntimeError(f"No tensor files found in {root_dir}")
        
        print(f"Found {len(self.tensor_files)} tensor files in total")
        
    
    def _extract_timestamp(self, filename):
        """Extract timestamp from filename as YYYYMMDD_HHMMSS_mmm when present."""
        # Remove extension
        base = os.path.splitext(filename)[0]
        # Find all parts separated by underscores
        parts = base.split('_')
        # First try full timestamp pattern found in files like grid_20260312_115711_924.pt
        for i in range(len(parts) - 2):
            p0, p1, p2 = parts[i], parts[i + 1], parts[i + 2]
            if p0.isdigit() and p1.isdigit() and p2.isdigit() and len(p0) == 8 and len(p1) == 6:
                return f"{p0}_{p1}_{p2}"

        # Fallback: first numeric part (supports legacy naming)
        for part in parts:
            try:
                float(part)
                return part
            except ValueError:
                continue
        raise ValueError(f"Could not extract timestamp from filename: {filename}")

    def _timestamp_to_key(self, timestamp):
        """Convert timestamp strings/numbers to a comparable integer key."""
        timestamp_str = str(timestamp).strip()
        digits = ''.join(ch for ch in timestamp_str if ch.isdigit())
        if not digits:
            raise ValueError(f"Could not convert timestamp to key: {timestamp}")
        return int(digits)
    
    def _is_timestamp_in_presence_interval(self, timestamp):
        """Check if a timestamp falls within any presence interval"""
        timestamp_key = self._timestamp_to_key(timestamp)
        for start, end in self.presence_intervals:
            if start <= timestamp_key <= end:
                return True
        return False
    
    def __len__(self):
        return len(self.tensor_files)

    def _pad_or_crop_tensor(self, tensor):
        """Pad with zeros or crop tensor to target (height, width) when configured."""
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor with shape (C, H, W), got {tuple(tensor.shape)}")

        channels, current_height, current_width = tensor.shape
        if channels != 2:
            raise ValueError(f"Expected 2 channels, got {channels} for tensor shape {tuple(tensor.shape)}")

        target_height = self.height if self.height is not None else current_height
        target_width = self.width if self.width is not None else current_width

        # Crop if larger than target
        tensor = tensor[:, :target_height, :target_width]

        # Pad with zeros if smaller than target
        pad_h = max(0, target_height - tensor.shape[1])
        pad_w = max(0, target_width - tensor.shape[2])
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return tensor
    
    def _cast_tensor_dtype(self, tensor):
        """Cast tensor to the specified data type if configured."""
        if self.data_type is not None:
            return tensor.to(self.data_type)
        return tensor

    def __getitem__(self, idx):
        nsyms = 30 * 6
        f = self.tensor_files[idx // nsyms]
        grid = loadmat(f)['dataSSB'][:,idx%6,idx%30]
        label = f.split('/')[-1].split('(')[-1].split(')')[0]

        print(f'{type(grid)=}')
        print(f'{grid.shape=}')
        t = torch.from_numpy(grid)
        print(t)

        # Get the FFT using the complex grid
        ffted = fft(grid)

        # Collapse phase and amplitude into the same channel as in paper
        real_imag = torch.cat((ffted.real, ffted.imag), 0)
        t = torch.reshape(real_imag, (1,2*540))

        return t, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GridCSVDataset")
    parser.add_argument('--dataset_path', type=str, help='Path to the directory containing grid CSV files')
    parser.add_argument('--labels_csv', type=str, help='Path to CSV file with timestamps and events')
    parser.add_argument('--concat_grids', type=int, default=1, help='Number of consecutive resource grids to concatenate')
    parser.add_argument('--presence_threshold', type=int, default=1, help='Minimum number of presence grids required to label a block as presence')
    parser.add_argument('--data_type', type=str, default='float32', help='Data type for tensors (e.g., float32, int64)')
    parser.add_argument('--height', type=int, default=540, help='Height to pad or crop tensors to')
    parser.add_argument('--width', type=int, default=14, help='Width to pad or crop tensors to')
    args = parser.parse_args()

    dataset = GridTensorDataset(args.dataset_path,
                             args.labels_csv,
                             concat_grids=args.concat_grids,
                             presence_grids_threshold=args.presence_threshold,
                             data_type=getattr(torch, args.data_type),
                             height=args.height,
                             width=args.width)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Classes: {dataset.classes}")
    random_idx = torch.randint(0, len(dataset), (1,)).item()
    print(f"Example sample shape: {dataset[random_idx][0].shape}, label: {dataset[random_idx][1]}")




