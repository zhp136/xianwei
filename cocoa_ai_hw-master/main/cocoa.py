import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
sys.path.append('../misc/')

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import h5py as hp
import sympy as sym
import matplotlib.animation as animation
import mrcfile
import tifffile
import argparse
from scipy.io import savemat, loadmat

from models import *
from utils import *
from losses import *

from scipy.signal import wiener
from psf_torch import PsfGenerator3D
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR, OneCycleLR, MultiStepLR, ExponentialLR
from PIL import Image
from skimage.transform import rescale, resize
from torch.fft import fftn, ifftn, fftshift, ifftshift, rfftn, irfftn, rfft

import random

seed = 100  # 你可以换成其他喜欢的数字
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dtype = torch.cuda.FloatTensor
torch.backends.cudnn.benchmark = True
gfp = custom_div_cmap(2**16-1, mincol='#000000', midcol = '#00FF00', maxcol='#FFFFFF')


parser = argparse.ArgumentParser(description="Hyperparameters - Beads")

parser.add_argument('--filepath_ref', type=str, default='../source/beads/')
parser.add_argument('--filepath', type=str, default='../source/beads/')

# 与文件读写有关的超参数
parser.add_argument('--net_obj_save_path_pretrained_prefix', type=str, default='../rec/')
parser.add_argument('--net_obj_save_path_pretrained_suffix', type=str, default='_skips_2468')
parser.add_argument('--net_obj_save_path_trained_prefix', type=str, default='../rec/')
parser.add_argument('--net_obj_save_path_trained_suffix', type=str, default='_skips_2468_full_trained')
parser.add_argument('--rec_save_path_prefix', type=str, default='../rec/')
parser.add_argument('--rec_save_path_suffix', type=str, default='enc_angle_3_depth_7_tvz_2em9_filters_128_skips_12')
parser.add_argument('--suffix', type=str, default='real_time')
parser.add_argument('--suffix_rec', type=str, default='rsd_reg')

# 与数据文件大小和系统物理参数有关的超参数
parser.add_argument('--cnts', type=list, default=[100, 224, 224])
parser.add_argument('--dims', type=list, default=[100, 224, 224])
parser.add_argument('--padding', type=int, default=24)
parser.add_argument('--normalized', type=bool, default=True)
parser.add_argument('--psf_dz', type=float, default=0.5)
parser.add_argument('--psf_dy', type=float, default=0.1)
parser.add_argument('--psf_dx', type=float, default=0.1)
parser.add_argument('--n_detection', type=float, default=1.1)
parser.add_argument('--emission_wavelength', type=float, default=0.515)
parser.add_argument('--n_obj', type=float, default=1.333)

# 与编码方式有关的超参数
parser.add_argument('--encoding_option', type=str, default='radial')
parser.add_argument('--radial_encoding_angle', type=float, default=3,
                    help='Typically, 3 ~ 7.5. Smaller values indicates the ability to represent fine features.')
parser.add_argument('--radial_encoding_depth', type=int, default=7,
                    help='If too large, stripe artifacts. If too small, oversmoothened features. Typically, 6 or 7.') # 7, 8 (jiggling artifacts)

# 与网络结构有关的超参数
parser.add_argument('--nerf_num_layers', type=int, default=6)
parser.add_argument('--nerf_num_filters', type=int, default=128) 
parser.add_argument('--nerf_skips', type=list, default=[2,4,6])
parser.add_argument('--nerf_beta', type=float, default=1.0)
parser.add_argument('--nerf_max_val', type=float, default=40.0)

# 与训练逻辑有关的超参数
parser.add_argument('--pretraining', type=bool, default=True)
parser.add_argument('--pretraining_num_iter', type=int, default=400) 
parser.add_argument('--pretraining_lr', type=float, default=1e-2)
parser.add_argument('--pretraining_measurement_scalar', type=float, default=5.) # > 1
parser.add_argument('--training_num_iter', type=int, default=1000)
parser.add_argument('--training_lr_obj', type=float, default=5e-3)
parser.add_argument('--training_lr_ker', type=float, default=1e-2)
parser.add_argument('--kernel_max_val', type=float, default=1e-2)
parser.add_argument('--kernel_order_up_to', type=int, default=4) 
parser.add_argument('--lr_schedule', type=str, default='cosine')

# 与损失函数有关的超参数
parser.add_argument('--ssim_weight', type=float, default=1.0)
parser.add_argument('--tv_z', type=float, default=1e-9,
                    help='larger tv_z helps for denser samples.') 
parser.add_argument('--tv_z_normalize', type=bool, default=False)
parser.add_argument('--rsd_reg_weight', type=float, default=5e-4,
                    help='Helps to retrieve aberrations correctly. Too large, skeletionize the image.')
 

args = parser.parse_args(args=[])

file_directory = "../source/ours/" # TODO: 替换为你的文件路径
file_name = "figure_211.tif" # TODO: 替换为你的文件路径
data = tifffile.imread(os.path.join(file_directory, file_name)).astype(np.float32)


data = data / data.max() # 归一化


data_shape = data.shape

y = torch.tensor(data.copy()).type(dtype).cuda(0).view(data.shape[0], data.shape[1], data.shape[2])

INPUT_HEIGHT = data.shape[1]
INPUT_WIDTH = data.shape[2]
INPUT_DEPTH = data.shape[0]

psf = PsfGenerator3D(psf_shape=(data.shape[0], data.shape[1], data.shape[2]), 
                             units=(args.psf_dz, args.psf_dy, args.psf_dx), 
                             na_detection=args.n_detection, 
                             lam_detection=args.emission_wavelength, 
                             n=args.n_obj) 

coordinates = input_coord_2d(INPUT_WIDTH, INPUT_HEIGHT).cuda(0)
coordinates = radial_encoding(coordinates, args.radial_encoding_angle, args.radial_encoding_depth).cuda(0) 


net_obj = NeRF(D = args.nerf_num_layers,
                W = args.nerf_num_filters,
                skips = args.nerf_skips, 
                in_channels = coordinates.shape[-1], 
                out_channels = INPUT_DEPTH).cuda(0)

net_obj_save_path_pretrained = args.net_obj_save_path_pretrained_prefix + 'net_obj_' + args.suffix + '_' +  args.net_obj_save_path_pretrained_suffix + '_' + str(args.radial_encoding_angle) + '_' + str(args.radial_encoding_depth) + '_' + str(args.nerf_num_filters) + '_' + str(np.sum(args.nerf_skips)) + '.pth'
net_obj_save_path_trained = args.net_obj_save_path_trained_prefix + 'net_obj_' + args.suffix + '_' +  args.net_obj_save_path_trained_suffix + '_' + str(args.radial_encoding_angle) + '_' + str(args.radial_encoding_depth) + '_' + str(args.nerf_num_filters) + '_' + str(np.sum(args.nerf_skips)) + '.pth'


if not os.path.exists(net_obj_save_path_pretrained) or args.pretraining:
    t_start = time.time()

    optimizer = torch.optim.Adam([{'params':net_obj.parameters(), 'lr':args.pretraining_lr}],
                                    betas=(0.9, 0.999), eps = 1e-8)
    if args.lr_schedule == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=[1000, 1500, 2000], gamma=0.5)
    elif args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.pretraining_num_iter, args.pretraining_lr/25)

    loss_list = np.empty(shape = (1 + args.pretraining_num_iter, ))
    loss_list[:] = np.nan

    for step in tqdm(range(args.pretraining_num_iter)):
        out_x = net_obj(coordinates)

        if args.nerf_beta is None:
            out_x = args.nerf_max_val * nn.Sigmoid()(out_x)
        else:
            out_x = nn.Softplus(beta = args.nerf_beta)(out_x)

        out_x_m = out_x.view(data.shape[1], data.shape[2], data.shape[0]).permute(2, 0, 1)

        loss = ssim_loss(out_x_m, args.pretraining_measurement_scalar * y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_list[step] = loss.item()

    t_end = time.time()
    print('Initialization - Elapsed time: ' + str(t_end - t_start) + ' seconds.')

else:
    net_obj.load_state_dict(torch.load(net_obj_save_path_pretrained)); print('Pre-trained model loaded.')
    net_obj.train()


## kernel with simple coefficients
net_ker = optimal_kernel(max_val = args.kernel_max_val, order_up_to = args.kernel_order_up_to) # 1e-2  4    12个

optimizer = torch.optim.Adam([{'params':net_obj.parameters(), 'lr':args.training_lr_obj},  # 1e-3
                                {'params':net_ker.parameters(), 'lr':args.training_lr_ker}], # 4e-3
                                betas = (0.9, 0.999), eps = 1e-8)

scheduler = CosineAnnealingLR(optimizer, args.training_num_iter, args.training_lr_ker/25)

loss_list = np.empty(shape = (1 + args.training_num_iter, )); loss_list[:] = np.nan
wfe_list = np.empty(shape = (1 + args.training_num_iter, )); wfe_list[:] = np.nan
lr_obj_list = np.empty(shape = (1 + args.training_num_iter, )); lr_obj_list[:] = np.nan
lr_ker_list = np.empty(shape = (1 + args.training_num_iter, )); lr_ker_list[:] = np.nan

t_start = time.time()

for step in tqdm(range(args.training_num_iter)):
    out_x = net_obj(coordinates)

    if args.nerf_beta is None:
        out_x = args.nerf_max_val * nn.Sigmoid()(out_x)
    else:
        out_x = nn.Softplus(beta = args.nerf_beta)(out_x)
        out_x = torch.minimum(torch.full_like(out_x, args.nerf_max_val), out_x) # 40.0

    out_x_m = out_x.view(data.shape[1], data.shape[2], data.shape[0]).permute(2, 0, 1)

    wf = net_ker.k

    out_k_m = psf.incoherent_psf(wf, normalized = args.normalized) / data.shape[0]
    k_vis = psf.masked_phase_array(wf)
    out_y = fft_convolve(out_x_m, out_k_m, mode='fftn')

    loss = args.ssim_weight * ssim_loss(out_y, y)

    loss += single_mode_control(wf, 1, -0.0, 0.0) # quite crucial for suppressing unwanted defocus. ？

    loss += args.tv_z * tv_1d(out_x_m, axis = 'z', normalize = args.tv_z_normalize)
    loss += args.rsd_reg_weight * torch.reciprocal(torch.std(out_x_m) / torch.mean(out_x_m)) # 4e-3

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.lr_schedule == 'cosine':
        scheduler.step()     

    elif args.lr_schedule == 'multi_step':
        if step == 500-1:
            optimizer.param_groups[0]['lr'] = args.training_lr_obj/10

        if step == 750-1:
            optimizer.param_groups[1]['lr'] = args.training_lr_ker/10
            optimizer.param_groups[0]['lr'] = args.training_lr_obj/100

    loss_list[step] = loss.item()
    wfe_list[step] = (args.emission_wavelength * 1e3 * torch.sqrt(torch.sum(torch.square(wf)))).detach().cpu().numpy()  # wave -> nm RMS
    lr_obj_list[step] = optimizer.param_groups[0]['lr']
    lr_ker_list[step] = optimizer.param_groups[1]['lr']


t_end = time.time()
print('Training - Elapsed time: ' + str(t_end - t_start) + ' seconds.')


y = y.detach().cpu().numpy()
out_x_m = out_x_m.detach().cpu().numpy()
out_k_m = out_k_m.detach().cpu().numpy()
out_y = out_y.detach().cpu().numpy()
wf = wf.detach().cpu().numpy()


hf = hp.File(args.rec_save_path_prefix + 'rec_' + args.suffix + '_' + '_normalized_' + str(args.normalized) + '_' + args.suffix_rec + '_num_iter_' + str(args.training_num_iter) + '_' + args.rec_save_path_suffix + '_' + str(args.nerf_num_filters) + '_' + str(np.sum(args.nerf_skips)) + '.h5', 'w')
hf.create_dataset('out_x_m', data=out_x_m[:, 
                                            args.padding:2*args.dims[1]+args.padding, 
                                            args.padding:2*args.dims[2]+args.padding])
hf.create_dataset('out_k_m', data=out_k_m)
hf.create_dataset('out_y', data=out_y)
hf.create_dataset('wf', data=wf)
hf.create_dataset('loss_list', data=loss_list)
hf.create_dataset('y', data=y)
# hf.create_dataset('y_min', data=y_min)
# hf.create_dataset('y_max', data=y_max)
hf.close()

f = open(args.rec_save_path_prefix + 'rec_' + args.suffix + '_' + '_normalized_' + str(args.normalized) + '_' + args.suffix_rec + '_num_iter_' + str(args.training_num_iter) + '_' + args.rec_save_path_suffix + '_' + str(args.nerf_num_filters) + '_' + str(np.sum(args.nerf_skips)) + '.txt', 'w')
f.write(str(vars(args)))
f.close()







wf_est = estimated_to_given_conversion_ml_to_dmd(wf, order_up_to = args.kernel_order_up_to)
wf_ref = wf_est

# Post-processing
th = out_x_m.mean() + 1.5 * out_x_m.std()
out_x_m = np.maximum(out_x_m, th)

out_x_m = out_x_m / out_x_m.max()

measurement_save = out_x_m * 65535
tifffile.imwrite(os.path.join(file_directory, "cocoa-"+file_name), measurement_save.astype(np.uint16))
savemat(os.path.join(file_directory, "cocoa-"+file_name.split(".")[0]+".mat"), {
    "cocoa_shape": data_shape,
    "voxel_unit": (args.psf_dz, args.psf_dy, args.psf_dx),
    "wave_length": args.emission_wavelength,
    "n": args.n_obj,
    "NA": args.n_detection,
    "wf": wf_ref,
})
