import torch
import torch.nn as nn
import argparse
from torch.fft import fft, ifft
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from models import transnet
from dataloader import Cost2100DataLoader, GridTensorDataset, GridSymbolTensorDataset
from tensorboardX import SummaryWriter
from torchviz import make_dot
from torch.utils.data import DataLoader
import glob


# https://discuss.pytorch.org/t/polar-coordinates-transformation-layer/66717/2
def to_polar(x, y):
    return (x**2 + y**2).sqrt(), torch.atan(y/x)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
        help='path to TransNet model')
    parser.add_argument('--cr', type=int,
                    help='compression ratio')
    parser.add_argument('--d_model', type=int,
                    help='d_model parameter')
    parser.add_argument('--dim_feedforward', type=int,
                    help='2*540 for GIROS')
    parser.add_argument('--grids_dir', type=str,
        help='path to dir with grids .pt')
    parser.add_argument('--out_dir', type=str,
                    help='path to store the compressed grids')
    args = parser.parse_args()


    # Load the TransNet model
    model = transnet(reduction=args.cr, d_model=args.d_model,
                     dim_feedforward=args.dim_feedforward)
    model.load_state_dict(torch.load(args.model_path,
        weights_only=False)['state_dict'])
    model.eval()


    # pass
    for f in glob.glob(f'{args.grids_dir}/*.pt'):
        fname = f.split('/')[-1]
        z = torch.zeros(2, 540, 14)
        t = torch.load(f).to(torch.float32)
        t_comp = torch.zeros(z.shape)

        # Fill with zero missing symbols
        z[:,:,:t.shape[-1]] = t
        t = z

        for sym in range(14):
            # FFT
            ffted = fft(torch.polar(t[0,:,sym], t[1,:,sym]))
            # Collapse phase and amplitude into the same channel as in paper
            real_imag = torch.cat((ffted.real, ffted.imag), 0)
            concatenated_tensor = torch.reshape(real_imag, (1,2*540))
            out = model(concatenated_tensor)[0,0,:]
            iffted = ifft(torch.complex(out[:540], out[540:]))
            ampl, phase = to_polar(iffted.real, iffted.imag)

            t_comp[0,:,sym] = ampl
            t_comp[1,:,sym] = phase

        print('Saving at: ', f'{args.out_dir}/{fname}')
        torch.save(t, f'{args.out_dir}/{fname}')


