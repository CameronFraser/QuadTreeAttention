import os
import argparse
import torch
import cv2
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from loftr.src.config.default import get_cfg_defaults
from loftr.src.utils.misc import lower_config

from loftr.src.loftr import LoFTR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QuadTreeAttention demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--weight', type=str, default=Path.cwd() / "weights/indoor.ckpt", help="Path to the checkpoint.")
    parser.add_argument('--config_path', type=str,
                        default=Path.cwd() / "FeatureMatching/loftr/configs/loftr/indoor/loftr_ds_quadtree.py", help="Path to the config.")
    parser.add_argument('--input_1st', type=str, help="1st image.")
    parser.add_argument('--input_2nd', type=str, help="2nd image.")
    parser.add_argument('--output_folder', type=str,
                        default=Path.cwd() / 'data' / 'output', help="Folder to save .mat and plot in")
    parser.add_argument("--ransac_reproj_thresh", type=float, default=0.5,
                        help="Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix.")
    parser.add_argument("--ransac_confidence", type=float, default=0.999,
                        help="Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level of confidence (probability) that the estimated matrix is correct. In the range 0..1 exclusive.")
    parser.add_argument("--ransac_max_iters", type=int, default=100000)

    opt = parser.parse_args()

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(opt.config_path)
    _config = lower_config(config)

    # Matcher: LoFTR
    matcher = LoFTR(config=_config['loftr'])
    state_dict = torch.load(opt.weight, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict, strict=True)

    # Load example images
    img0_pth = opt.input_1st
    img1_pth = opt.input_2nd
    img0_raw = cv2.imread(img0_pth)
    img1_raw = cv2.imread(img1_pth)

    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0_raw_gs = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2GRAY)
    img1_raw_gs = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2GRAY)

    img0_raw_color = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB)
    img1_raw_color = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2RGB)

    img0 = torch.from_numpy(img0_raw_gs)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw_gs)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher.eval()
        matcher.to('cuda')
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
        mask_conf = mconf > 0
        conf = mconf[mask_conf]

    Fm, inliers = cv2.findFundamentalMat(
        mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    dict_to_save = {}
    dict_to_save['kpt1'] = mkpts0
    dict_to_save['kpt2'] = mkpts1
    dict_to_save['conf'] = mconf
    dict_to_save['inliers'] = inliers

    print(f"Found {len(mkpts0)} matches")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                     torch.ones(mkpts0.shape[0]).view(
                                         1, -1, 1, 1),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                     torch.ones(mkpts1.shape[0]).view(
                                         1, -1, 1, 1),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        img0_raw_color,
        img1_raw_color,
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None,
                   'feature_color': (0.2, 0.5, 1), 'vertical': False}
    )

    plt.title(
        f"Raw matches {len(mkpts0)}, inliers {len(inliers[inliers == True])}")

    output_dir = Path(opt.output_folder)
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / 'output.png')

    sio.savemat(output_dir / 'matches.mat', dict_to_save)
