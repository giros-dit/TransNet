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
    parser.add_argument('-fawn', action='store_true',
                    help='flag for FAWN dataset')
    parser.add_argument('-fraunhofer', action='store_true',
                    help='flag for Fraunhofer dataset')
    args = parser.parse_args()


    # Load the TransNet model
    print(f'{args.dim_feedforward=}')
    if args.fawn:
        out_dim = 2*360
    elif args.fawn:
        out_dim = 2*(49+5)
    else:
        out_dim = 2*540
    model = transnet(reduction=args.cr, d_model=args.d_model,
                     dim_feedforward=args.dim_feedforward,
                     out_dim=out_dim)
    model.load_state_dict(torch.load(args.model_path,
        weights_only=False)['state_dict'])
    model.eval()


    # pass
    for f in glob.glob(f'{args.grids_dir}/*.pt'):
        fname = f.split('/')[-1]
        if args.fawn:
            z = torch.zeros(2, 360, 6)
        elif args.fraunhofer:
            z = torch.zeros(2, 49+5, 6)
        else:
            z = torch.zeros(2, 540, 14)
        t = torch.load(f).to(torch.float32)
        t_comp = torch.zeros(z.shape)

        # Fill with zero missing symbols
        if args.fraunhofer:
            z[:,:49,:t.shape[-1]] = t
        else:
            z[:,:,:t.shape[-1]] = t
        t = z

        if args.fawn or args.fraunhofer:
            nsyms = 6
        else:
            nsyms = 14
        for sym in range(nsyms):
            # FFT
            ffted = fft(torch.polar(t[0,:,sym], t[1,:,sym]))
            # Collapse phase and amplitude into the same channel as in paper
            real_imag = torch.cat((ffted.real, ffted.imag), 0)
            if args.fawn:
                concatenated_tensor = torch.reshape(real_imag, (1,2*360))
            elif args.fraunhofer:
                concatenated_tensor = torch.reshape(real_imag, (1,2*(49+5)))
            else:
                concatenated_tensor = torch.reshape(real_imag, (1,2*540))
            out = model(concatenated_tensor)[0,0,:]
            if args.fawn:
                iffted = ifft(torch.complex(out[:360], out[360:]))
            elif args.fraunhofer:
                iffted = ifft(torch.complex(out[:49+5], out[49+5:]))
            else:
                iffted = ifft(torch.complex(out[:540], out[540:]))
            ampl, phase = to_polar(iffted.real, iffted.imag)

            t_comp[0,:,sym] = ampl
            t_comp[1,:,sym] = phase
            
        # Remove the extra padding added for TransNet input
        if args.fraunhofer:
            t_comp = ampl[:,:49,:]

        print('Saving at: ', f'{args.out_dir}/{fname}')
        torch.save(t_comp, f'{args.out_dir}/{fname}')


