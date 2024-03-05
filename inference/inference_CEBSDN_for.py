import argparse
import time
import cv2
import glob
import numpy as np
import os
import torch
from torchinfo import summary
# from torchstat import stat
from ptflops import get_model_complexity_info
from thop import profile
from tqdm import tqdm
import time
from basicsr.archs.CEBSDN_arch import CEBSDN as model
from basicsr.archs.CEBSDN_ex_arch import CEBSDN_wo_CA,CEBSDN_wo_Cat
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

if __name__ == '__main__':
    model_name = 'CEBSDN'
    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='datasets/Helen/Helen_test')
    # parser.add_argument('--test_path', type=str, default='datasets/data/inference_test')
    parser.add_argument(
        '--model_path_list',
        type=str,
        default=  # noqa: E251
        'experiments\\infer')
    parser.add_argument(
        '--model_num',
        type=str,
        default=  # noqa: E251
        'net_g_5000.pth')
    args = parser.parse_args()
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    test_root = os.path.join(args.test_path)
    test_HR = os.path.join(test_root,"HR")
    test_LR = os.path.join(test_root,"LR")
    model_path_list = os.listdir(args.model_path_list)
    psnrlist = []
    psnr_y_list=[]
    ssimlist = []
    runtimelist = []
    for model_name_path in model_path_list:
        model_path = os.path.join(args.model_path_list,model_name_path,"models",args.model_num)
        if not os.path.exists(model_path):
            print(f'{model_name_path} no model {args.model_num} break')
            break
        if 'CEBSDN_Helenx8_C48BS64_infer_L1_100k' == model_name_path:
            net = model(num_feat=48,upscale=8).to(device)
        elif 'CEBSDN_Helenx8_C58BS64_infer_L1_100k' == model_name_path:
            net = model(num_feat=58,upscale=8).to(device)
        elif 'CEBSDN_Helenx8_C62BS64_infer_L1_100k' == model_name_path:
            net = model(num_feat=62,upscale=8).to(device)
        elif 'CEBSDN_Helenx8_C64BS64_infer_L1_100k' == model_name_path:
            net = model(num_feat=64,upscale=8).to(device)
        elif 'CEBSDN_wo_CA_Helenx8_C48BS64_L1_infer_100k' == model_name_path:
            net = CEBSDN_wo_CA(num_feat=48,upscale=8).to(device)
        elif 'CEBSDN_wo_Cat_Helenx8_C48BS64_L1_infer_100k' == model_name_path:
            net = CEBSDN_wo_Cat(num_feat=48,upscale=8).to(device)
        elif 'CEBSDN_wo_ECA_Helenx8_C48BS64_infer_100k' == model_name_path:
            net = CEBSDN_wo_CA(num_feat=48,upscale=8).to(device)
        elif 'CEBSDN_wo_ECA_Helenx8_C48BS64_infer_10k' == model_name_path:
            continue

        # result
        result_root = f'results/fsr_result/{args.model_num}_infer/{model_name_path}'
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

        inp = torch.randn(1, 3, 16, 16)
        flops, params = profile(net,inputs=(inp,) )
        print(f'Network: {model_name_path}, with flops(128 x 128): {flops/1e9:.2f} GMac, with active parameters: {params/1e3} K.')
        from thop import clever_format
        macs, params = clever_format([flops, params], "%.3f")
        print(macs,params)

    # macs, params = get_model_complexity_info(net, (3, 16, 16), as_strings=True,
    #                                         print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # # scan all the jpg and png images
    # img_list = sorted(glob.glob(os.path.join(test_LR, '*.[jp][pn]g')))
    # pbar = tqdm(total=len(img_list), desc='')
    # for idx, img_path in enumerate(img_list):
    #     img_name_ex = os.path.basename(img_path)
    #     img_name = img_name_ex.split('.')[0]
    #     img_hr = cv2.imread(os.path.join(test_HR,img_name_ex), cv2.IMREAD_COLOR)

    #     pbar.update(1)
    #     pbar.set_description(f'{idx}: {img_name}')

    #     # read image
    #     img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    #     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    #     img = img.unsqueeze(0).to(device)
    #     # inference
    #     with torch.no_grad():
    #         # HR_2x,HR_4x,output,fb_sr2,fb_sr4,fb_sr8 = net(img)
    #         start = time.time()
    #         output = net(img)
    #         end =time.time()
    #         torch.cuda.synchronize()
    #         runtimelist.append(end-start)  # milliseconds
    #     # save image
    #     # output = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
    #     output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    #     if output.ndim == 3:
    #         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    #     output = (output * 255.0).round().astype(np.uint8)
    #     psnr_val = calculate_psnr(output,img_hr,8)
    #     psnr_y_val = calculate_psnr(output,img_hr,8,test_y_channel=True)
    #     ssim_val = calculate_ssim(output,img_hr,8)
    #     psnrlist.append(psnr_val)
    #     psnr_y_list.append(psnr_y_val)
    #     ssimlist.append(ssim_val)
    #     save_img_path = os.path.join(result_root, f'{img_name}_fsr_sr8.png')
    #     cv2.imwrite(save_img_path, output)
    # ave_runtime = round(sum(runtimelist) / len(runtimelist) / 1000.0 , 6)
    # print(f'Ave psnr:{np.mean(psnrlist).round(4)} Ave ypsnr:{np.mean(psnr_y_list).round(4)}\nAve ssim:{np.mean(ssimlist).round(4)} ave_runtime:{ave_runtime}')