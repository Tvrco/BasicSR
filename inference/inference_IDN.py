import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torchinfo import summary
# from torchstat import stat
# from ptflops import get_model_complexity_info
from thop import profile
from tqdm import tqdm
from lpips import LPIPS
import time
from basicsr.archs.Idn_arch import IDN as model
# from basicsr.archs.BSRN_arch import BSRN as model
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

if __name__ == '__main__':
    model_name = 'IDN'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_path', type=str, default='datasets/Helen/Helen_test')
    # parser.add_argument('--test_path', type=str, default='datasets/data/inference_test')
    parser.add_argument('--test_path', type=str, default='datasets/data/inference_test')
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/IDN_CelebA_x8_C64B64_L1_1000k/models/net_g_200000.pth')
    args = parser.parse_args()
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    test_root = os.path.join(args.test_path)
    test_HR = os.path.join(test_root,"HR")
    test_LR = os.path.join(test_root,"LR_X8")
    psnrlist = []
    psnr_y_list=[]
    ssimlist = []
    runtimelist = []
    lpips_calculator = LPIPS(net='vgg')  # 使用默认的VGG模型，也可以根据需要选择其他模型
    lpips_calculator.to(device)  # 将LPIPS计算器移动到与模型相同的设备
    lpips_scores = []  # 用于存储LPIPS分数
    # result
    result_root = f'results/fsr_result/{model_name}_100K'
    os.makedirs(result_root, exist_ok=True)
    print(f"result_root:{result_root}\nbasename:{os.path.basename(args.test_path)}")
    # set up the LapSrnMSV
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    net = model(in_channels=3,out_channels=3, upscale=8).to(device)
    # net = model(upscale=8).to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['params'])
    net.eval()
    # summary(net,(1,3,16,16))
    # stat(net,(1,3,16,16))
    inp = torch.randn(1, 3, 16, 16).to(device)
    flops, params = profile(net,inputs=(inp,) )
    print("FLOPs=", str(flops/1e9) + '{}'.format("G"))
    print("params=", str(params/1e6) + '{}'.format("M"))

    # macs, params = get_model_complexity_info(net, (3, 16, 16), as_strings=True,
    #                                         print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # scan all the jpg and png images
    img_list = sorted(glob.glob(os.path.join(test_LR, '*.[jp][pn]g')))
    pbar = tqdm(total=len(img_list), desc='')
    for idx, img_path in enumerate(img_list):
        img_name_ex = os.path.basename(img_path)
        img_name = img_name_ex.split('.')[0]
        img_hr = cv2.imread(os.path.join(test_HR,img_name_ex), cv2.IMREAD_COLOR)

        pbar.update(1)
        pbar.set_description(f'{idx}: {img_name}')

        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            # HR_2x,HR_4x,output,fb_sr2,fb_sr4,fb_sr8 = net(img)
            start.record()
            # start = time.time()
            output = net(img)
            # end = time.time()
            end.record()
            torch.cuda.synchronize()
            # runtimelist.append(end-start)  # milliseconds
            runtimelist.append(start.elapsed_time(end))  # milliseconds
                                # 计算LPIPS分数
            img_hr_tensor = img2tensor(img_hr).to(device)
            lpips_score = lpips_calculator(output, img_hr_tensor)
            lpips_scores.append(lpips_score.item())
        # save image
        # output = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        psnr_val = calculate_psnr(output,img_hr,4)
        psnr_y_val = calculate_psnr(output,img_hr,4,test_y_channel=True)
        ssim_val = calculate_ssim(output,img_hr,4)
        psnrlist.append(psnr_val)
        psnr_y_list.append(psnr_y_val)
        ssimlist.append(ssim_val)
        save_img_path = os.path.join(result_root, f'{img_name}_fsr_sr8.png')
        cv2.imwrite(save_img_path, output)
    ave_runtime = sum(runtimelist) / len(runtimelist)
    Fps_time = 1000/ ave_runtime
    print(f'Ave psnr:{np.mean(psnrlist).round(4)} Ave ypsnr:{np.mean(psnr_y_list).round(4)}\nAve ssim:{np.mean(ssimlist).round(4)} ave_runtime:{ave_runtime} Fps_time:{Fps_time}')
    # ave_lpips_score = np.mean(lpips_scores).round(4)
    # print(f'Ave LPIPS:{ave_lpips_score}')