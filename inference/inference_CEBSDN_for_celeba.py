import argparse
import time
import cv2
import glob
import numpy as np
import os
import sys
import torch
from torchinfo import summary
# from torchstat import stat
from ptflops import get_model_complexity_info
from thop import profile
from tqdm import tqdm
import time
import logging
from basicsr.archs.BSRN_arch import BSRN
from basicsr.archs.CEBSDN_arch import CEBSDN as model
from basicsr.archs.CEBSDN_ex_arch import CEBSDN_wo_CA,CEBSDN_wo_Cat
from basicsr.archs.Idn_arch import IDN
from basicsr.archs.PAN_arch import PAN
from basicsr.archs.RFDN_arch import RFDN
from basicsr.archs.VAPSR_arch import vapsr
from basicsr.archs.lkdn_arch import LKDN
from basicsr.archs.rlfn_arch import RLFN
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
def pretty_print_performance(time_dic):
    header = f"{'Model Name':<50} {'Mean Time (s)':<15} {'FPS':<10} {'Parameters':<20} {'macs':<20}"
    divider = '-' * len(header)
    print(header)
    print(divider)
    for model_name, timings in time_dic.items():
        mean_time = timings['mean_time']
        fps = timings['FPS']
        parameters = timings['parameters']
        macs = timings['macs']
        print(f"{model_name:<50} {mean_time:<15.4f} {fps:<10.2f} {parameters:<20} {macs:<20}")

if __name__ == '__main__':
    model_name = 'CEBSDN_CelebA-HQ'
    # device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_path', type=str, default='datasets/Helen/Helen_test')
    parser.add_argument('--test_path', type=str, default='datasets/data/inference_test')
    parser.add_argument(
        '--model_path_list',
        type=str,
        default=  # noqa: E251
        'experiments\\celeba')
    parser.add_argument(
        '--model_num',
        type=str,
        default=  # noqa: E251
        'net_g_latest.pth')
    args = parser.parse_args()
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    test_root = os.path.join(args.test_path)
    test_HR = os.path.join(test_root,"HR")
    test_LR = os.path.join(test_root,"LR_X8")
    model_path_list = os.listdir(args.model_path_list)
    psnrlist = []
    psnr_y_list=[]
    ssimlist = []
    runtimelist = []
    time_dic = {}
    for model_name_path in model_path_list:
        model_path = os.path.join(args.model_path_list,model_name_path,"models",args.model_num)
        if not os.path.exists(model_path):
            print(f'{model_name_path} no model {args.model_num} break')
            break
        if 'BSRN' in model_name_path:
            net = BSRN(num_feat=64,upscale=8).to(device)
        elif 'IDN_CelebA_x8_C64B64_L1_1000k' == model_name_path:
            net = IDN(in_channels=3,out_channels=3, upscale=8).to(device)
        elif 'LKDN_CelebA_x8_C64B64_L1_1000k' == model_name_path:
            net = LKDN(num_in_ch=3,
                        num_out_ch=3,
                        num_feat=56,
                        num_atten=56,
                        num_block=8,
                        upscale=8,
                        num_in=4,
                        conv='BSConvU',
                        upsampler='pixelshuffledirect').to(device)
        elif 'PAN_CelebA_x8_C40B64_L1_1000k' == model_name_path:
            net = PAN(in_nc=3,
                        out_nc=3,
                        nf=40,
                        unf=40,
                        nb=16,
                        scale=8).to(device)
        elif 'RFDN_CelebA_x8_C64B64_L1_1000k' == model_name_path:
            net = RFDN(in_nc=3,
                        nf=50,
                        num_modules=4,
                        out_nc=3,
                        upscale=8).to(device)
        elif 'RLFN_CelebA_x8_C64B64_L1_1000k' == model_name_path:
            net = RLFN(in_channels=3,
                        out_channels=3,
                        feature_channels=48,
                        mid_channels=48,
                        upscale_factor=8).to(device)
        elif 'VapSR_CelebA_x8_C64B64_L1_1000k' == model_name_path:
            net = vapsr(num_in_ch=3,
                                num_feat=48,
                                d_atten=64,
                                num_block=21,
                                num_out_ch=3,
                                scale=8).to(device)
        elif 'CEBSDN_CelebAx8_C48BS64_L1_600k' == model_name_path:
            net = model(num_feat=48,upscale=8).to(device)
        elif 'CEBSDN_wo_ECA_Helenx8_C48BS64_infer_10k' == model_name_path:
            continue

        # result
        result_root = f'results/fsr_result/{model_name}_{args.model_num}_infer/{model_name_path}'
            # set up the LapSrnMSV
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        os.makedirs(result_root, exist_ok=True)
        print(f"result_root:{result_root}\nbasename:{os.path.basename(args.test_path)}")
        # set up the LapSrnMSV
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['params'])
        # net.eval()
        # summary(net,(1,3,16,16))
        # stat(net,(1,3,16,16))

        inp = torch.randn(1, 3, 16, 16).to(device)
        with suppress_stdout():
            flops, params = profile(net,inputs=(inp,) )
            # print(f'Network: {model_name_path}, with flops(128 x 128): {flops/1e9:.2f} GMac, with active parameters: {params/1e3} K.')
            from thop import clever_format
            macs, params = clever_format([flops, params], "%.4f")
        # Restore the original logging level
        # print(macs,params)

    # macs, params = get_model_complexity_info(net, (3, 16, 16), as_strings=True,
    #                                         print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # scan all the jpg and png images
        img_list = sorted(glob.glob(os.path.join(test_LR, '*.[jp][pn]g')))
        # pbar = tqdm(total=len(img_list), desc='')
        for idx, img_path in enumerate(img_list):
            img_name_ex = os.path.basename(img_path)
            img_name = img_name_ex.split('.')[0]
            img_hr = cv2.imread(os.path.join(test_HR,img_name_ex), cv2.IMREAD_COLOR)

            # pbar.update(1)
            # pbar.set_description(f'{idx}: {img_name}')

            # read image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
            # inference
            with torch.no_grad():
                # HR_2x,HR_4x,output,fb_sr2,fb_sr4,fb_sr8 = net(img)
                # start = time.time()
                start.record()
                output = net(img)
                end.record()
                # end =time.time()
                torch.cuda.synchronize()
                # runtimelist.append(end-start)  # milliseconds
                runtimelist.append(start.elapsed_time(end))
            # save image
            # output = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            psnr_val = calculate_psnr(output,img_hr,8)
            psnr_y_val = calculate_psnr(output,img_hr,8,test_y_channel=True)
            ssim_val = calculate_ssim(output,img_hr,8)
            psnrlist.append(psnr_val)
            psnr_y_list.append(psnr_y_val)
            ssimlist.append(ssim_val)
            save_img_path = os.path.join(result_root, f'{img_name}_fsr_sr8.png')
            cv2.imwrite(save_img_path, output)
        mean_time = sum(runtimelist) / len(runtimelist)
        Fps_tims = 1000 / mean_time
        time_dic[model_name_path] = {'mean_time': mean_time, 'FPS': Fps_tims, 'parameters': params, 'macs': macs}
        print(f'Ave psnr:{np.mean(psnrlist).round(4)} Ave ypsnr:{np.mean(psnr_y_list).round(4)}\nAve ssim:{np.mean(ssimlist).round(4)} ave_runtime:{mean_time} FPS:{Fps_tims}')
        print('-'*50)
    pretty_print_performance(time_dic)