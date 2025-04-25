import argparse

def init_opts():
    parser = argparse.ArgumentParser(description="Hyperparameters - Beads")

    parser.add_argument('--filepath_ref', type=str, default='../source/beads/')
    parser.add_argument('--filepath', type=str, default='../source/beads/')

    parser.add_argument('--net_obj_save_path_pretrained_prefix', type=str, default='../rec/')
    parser.add_argument('--net_obj_save_path_pretrained_suffix', type=str, default='_skips_2468')
    parser.add_argument('--net_obj_save_path_trained_prefix', type=str, default='../rec/')
    parser.add_argument('--net_obj_save_path_trained_suffix', type=str, default='_skips_2468_full_trained')
    parser.add_argument('--rec_save_path_prefix', type=str, default='../rec/')
    parser.add_argument('--rec_save_path_suffix', type=str, default='enc_angle_3_depth_7_tvz_2em9_filters_128_skips_12')

    parser.add_argument('--suffix', type=str, default='real_time')
    parser.add_argument('--suffix_rec', type=str, default='rsd_reg')

    parser.add_argument('--cnts', type=list, default=[100, 224, 224])
    parser.add_argument('--dims', type=list, default=[100, 224, 224])
    parser.add_argument('--padding', type=int, default=24)
    parser.add_argument('--normalized', type=bool, default=True)
    parser.add_argument('--psf_dz', type=float, default=0.2)
    parser.add_argument('--psf_dy', type=float, default=0.086)
    parser.add_argument('--psf_dx', type=float, default=0.086)
    parser.add_argument('--n_detection', type=float, default=1.1)
    parser.add_argument('--emission_wavelength', type=float, default=0.515)
    parser.add_argument('--n_obj', type=float, default=1.333)

    parser.add_argument('--encoding_option', type=str, default='radial')
    parser.add_argument('--radial_encoding_angle', type=float, default=3,
                        help='Typically, 3 ~ 7.5. Smaller values indicates the ability to represent fine features.')
    parser.add_argument('--radial_encoding_depth', type=int, default=7,
                        help='If too large, stripe artifacts. If too small, oversmoothened features. Typically, 6 or 7.') # 7, 8 (jiggling artifacts)

    parser.add_argument('--nerf_num_layers', type=int, default=6)
    parser.add_argument('--nerf_num_filters', type=int, default=128) 
    parser.add_argument('--nerf_skips', type=list, default=[2,4,6])
    parser.add_argument('--nerf_beta', type=float, default=1.0)
    parser.add_argument('--nerf_max_val', type=float, default=40.0)

    parser.add_argument('--pretraining', type=bool, default=True)
    parser.add_argument('--pretraining_num_iter', type=int, default=400) 
    parser.add_argument('--pretraining_lr', type=float, default=1e-2)
    parser.add_argument('--pretraining_measurement_scalar', type=float, default=5.) # > 1
    parser.add_argument('--training_num_iter', type=int, default=1000)
    parser.add_argument('--training_lr_obj', type=float, default=5e-3)
    parser.add_argument('--training_lr_ker', type=float, default=1e-2)
    parser.add_argument('--kernel_max_val', type=float, default=1e-2)
    parser.add_argument('--kernel_order_up_to', type=int, default=4) 

    parser.add_argument('--ssim_weight', type=float, default=1.0)
    parser.add_argument('--tv_z', type=float, default=1e-9,
                        help='larger tv_z helps for denser samples.') 
    parser.add_argument('--tv_z_normalize', type=bool, default=False)
    parser.add_argument('--rsd_reg_weight', type=float, default=5e-4,
                        help='Helps to retrieve aberrations correctly. Too large, skeletionize the image.')

    parser.add_argument('--lr_schedule', type=str, default='cosine') 

    args = parser.parse_args(args=[])

    return args