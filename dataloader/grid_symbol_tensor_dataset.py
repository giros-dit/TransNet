"""
Author: Pablo Fernández López
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

class GridSymbolTensorDataset(Dataset):
    """Custom Dataset for loading and symbols from grids as tensors with external label CSV"""
    
    def __init__(self, root_dir, labels_csv_path, concat_grids=1, presence_grids_threshold=1,
                 data_type=torch.float32, height=None, width=None):
        """
        Args:
            root_dir: Path to the directory containing all tensor files
            labels_csv_path: Path to CSV file with timestamps and events (person_detected, person_lost)
            concat_grids: Number of consecutive resource grids to concatenate (default: 1)
            presence_grids_threshold: Minimum number of presence grids required to label
                a concatenated block as presence (default: 1)
            data_type: Data type for tensors (e.g., float32, int64)
            height: Height to pad or crop tensors to (default: None)
            width: Width to pad or crop tensors to (default: None)
        """
        self.root_dir = root_dir
        self.labels_csv_path = labels_csv_path
        self.concat_grids = concat_grids
        self.presence_grids_threshold = presence_grids_threshold
        self.height = height
        self.width = width
        self.data_type = data_type
        self.classes = ['no_presence', 'presence']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        if self.presence_grids_threshold < 1:
            raise ValueError("presence_grids_threshold must be >= 1")
        if self.presence_grids_threshold > self.concat_grids:
            raise ValueError(
                f"presence_grids_threshold ({self.presence_grids_threshold}) cannot be greater "
                f"than concat_grids ({self.concat_grids})"
            )
        
        # Load labels CSV and build presence intervals
        print(f"Loading labels from {labels_csv_path}")
        labels_df = pd.read_csv(labels_csv_path)
        labels_df['timestamp'] = labels_df['timestamp'].astype(str)
        labels_df = labels_df.sort_values('timestamp').reset_index(drop=True)
        
        # Build presence intervals from person_detected and person_lost events
        self.presence_intervals = []
        interval_start = None
        
        for _, row in labels_df.iterrows():
            if row['event'] == 'person_detected' and interval_start is None:
                interval_start = row['timestamp']
            elif row['event'] == 'person_lost' and interval_start is not None:
                self.presence_intervals.append(
                    (self._timestamp_to_key(interval_start), self._timestamp_to_key(row['timestamp']))
                )
                interval_start = None
        
        # If there's an open interval at the end, close it with the last timestamp
        if interval_start is not None:
            last_timestamp = labels_df.iloc[-1]['timestamp']
            self.presence_intervals.append(
                (self._timestamp_to_key(interval_start), self._timestamp_to_key(last_timestamp))
            )
            print(f"Warning: Open interval closed with last timestamp {last_timestamp}")
        
        print(f"Found {len(self.presence_intervals)} presence intervals")
        if self.presence_intervals:
            print(f"  Example intervals: {self.presence_intervals[:3]}")
        
        # Collect all tensor files from root directory
        tensor_files = sorted(glob.glob(os.path.join(root_dir, '*.pt')))
        if len(tensor_files) == 0:
            raise RuntimeError(f"No tensor files found in {root_dir}")
        
        print(f"Found {len(tensor_files)} tensor files in total")
        
        # Discard extra files that don't fit into complete blocks
        num_blocks = len(tensor_files) // concat_grids
        total_files_used = num_blocks * concat_grids
        tensor_files = tensor_files[:total_files_used]
        
        if total_files_used < len(tensor_files) + (len(tensor_files) % concat_grids):
            discarded = (len(tensor_files) + (len(tensor_files) % concat_grids)) - total_files_used
            print(f"Discarded {discarded} files to make complete blocks")
        
        print(f"Using {total_files_used} files grouped into {num_blocks} blocks of {concat_grids} resource grids")
        print(f"Presence threshold: {self.presence_grids_threshold}/{self.concat_grids} grids")
        
        # Group files into blocks and determine labels
        self.samples = []
        for i in range(0, len(tensor_files), concat_grids):
            block_files = tensor_files[i:i+concat_grids]
            
            # Count how many grids in this block fall within presence intervals.
            presence_count = 0
            for tensor_file in block_files:
                timestamp = self._extract_timestamp(os.path.basename(tensor_file))
                if self._is_timestamp_in_presence_interval(timestamp):
                    presence_count += 1
            
            label = (self.class_to_idx['presence']
                     if presence_count >= self.presence_grids_threshold
                     else self.class_to_idx['no_presence'])
            self.samples.append((block_files, label))
        
        print(f"Created {len(self.samples)} samples:")
        for class_name in self.classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            print(f"  {class_name}: {count} blocks")
    
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
        return len(self.samples)

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
        block_files, label = self.samples[idx // 14]
        
        # Load and convert each tensor file in the block to a tensor
        tensors = []
        for tensor_path in block_files:
            tensor = torch.zeros(2, 540, 14)
            # tensor = torch.load(tensor_path)
            loaded = torch.load(tensor_path)
            tensor[:loaded.shape[0],:loaded.shape[1],:loaded.shape[2]] = loaded
            tensor = self._pad_or_crop_tensor(tensor)
            tensor = self._cast_tensor_dtype(tensor)
            tensors.append(tensor[:,:,idx%14])
        
        # Concatenate tensors along the width dimension (dim=2)
        # Each tensor has shape (2, height, width), we concatenate along width
        #concatenated_tensor = torch.cat(tensors, dim=2)

        # Get the FFT using complex number with polar coordinates
        # amplitude (absolute), phase (angle)
        ffted = fft(torch.polar(tensors[0][0], tensors[0][1]))

        # Collapse phase and amplitude into the same channel as in paper
        real_imag = torch.cat((ffted.real, ffted.imag), 0)
        concatenated_tensor = torch.reshape(real_imag, (1,2*540))

        
        return concatenated_tensor, label


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
