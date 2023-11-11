import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from basicsr.archs.LapSRNV4_v3_3esdb_uptunel3_arch import LapSrnMSV4_12
from basicsr.utils.img_util import img2tensor, tensor2img

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='BasicSR/datasets/data/inference')
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/LapSRNV4.11_Celeb26k_BS64_L1_600k/models/net_g_200000.pth')
    args = parser.parse_args()
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    test_root = os.path.join(args.test_path)
    result_root = f'results/fbsr_result/{os.path.basename(args.test_path)}'
    os.makedirs(result_root, exist_ok=True)

    # set up the LapSrnMSV
    net = LapSrnMSV4_12(num_out_ch=3,dim=64).to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['params'])
    # net.eval()

    # # scan all the jpg and png images
    # img_list = sorted(glob.glob(os.path.join(test_root, '*.[jp][pn]g')))
    # pbar = tqdm(total=len(img_list), desc='')
    # for idx, img_path in enumerate(img_list):
    #     img_name = os.path.basename(img_path).split('.')[0]
    #     pbar.update(1)
    #     pbar.set_description(f'{idx}: {img_name}')
    #     # read image
    #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    #     # inference
    #     with torch.no_grad():
    #         HR_2x,HR_4x,output,fb_sr2,fb_sr4,fb_sr8 = net(img)
    #     # save image
    #     output = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
    #     save_img_path = os.path.join(result_root, f'{img_name}_bfsr_sr8.png')
    #     cv2.imwrite(save_img_path, output)
    #     print(fb_sr8.shape)
    #     reshaped_tensor  = fb_sr8.view(11, 1, 128,128)
    #     torchvision.utils.save_image(reshaped_tensor, f'tmp/fb_{i:03d}_{a}_{b}.png', nrow=11)
