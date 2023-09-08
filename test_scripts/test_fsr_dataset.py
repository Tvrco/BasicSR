import math
import os
from sympy import false
import torchvision.utils

from basicsr.data import build_dataloader, build_dataset

def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'celeba'
    opt['type'] = 'BFSRDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = 'D:\\data\\CelebA-HQ-img\\celeb128'
        opt['dataroot_lq16'] = 'D:\\data\\CelebA-HQ-img\\donwsamp\\celeb_16'
        opt['dataroot_lq32'] = 'D:\\data\\CelebA-HQ-img\\donwsamp\\celeb_32'
        opt['dataroot_lq64'] = 'D:\\data\\CelebA-HQ-img\\donwsamp\\celeb_64'
        opt['faceb_gt'] = 'D:\\data\\CelebA-HQ-img\\celeb_fb128_wight2'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')

    # elif mode == 'meta_info_file':
    #     opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
    #     opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    #     opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'  # noqa:E501
    #     opt['filename_tmpl'] = '{}'
    #     opt['io_backend'] = dict(type='disk')
    # elif mode == 'lmdb':
    #     opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
    #     opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'  # noqa:E501
    #     opt['io_backend'] = dict(type='lmdb')

    opt['gt_size'] = 128
    opt['use_hflip'] = False
    opt['use_rot'] = False

    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    dataset.__getitem__(0)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(f"批次{i}")

        lq = data['lq']
        lq32 = data['lq32']
        lq64 = data['lq64']
        gt = data['gt']
        fb_32 = data['fb_32']
        fb_64 = data['fb_64']
        fb_128 = data['fb_128']
        lq_path = data['lq_path']
        lq32_path = data['lq32_path']
        lq64_path = data['lq64_path']
        gt_path = data['gt_path']
        fb_path = data['fb_path']
        # print(lq_path,'\n',lq32_path,'\n',lq64_path,'\n', gt_path, '\n')
        # print(lq_path,'\n',lq32_path,'\n',lq64_path,'\n', gt_path, '\n', fb_path)
        # torchvision.utils.save_image(lq, f'tmp/lq_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(lq32, f'tmp/lq32_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(lq64, f'tmp/lq64_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        import matplotlib.pyplot as plt
        from torchvision import transforms
        t = transforms.ToPILImage()
        a,b = fb_128.shape[2],fb_128.shape[3]
        reshaped_tensor  = fb_128.view(16 * 11, 1, a,b)
        torchvision.utils.save_image(reshaped_tensor, f'tmp/fb_{i:03d}_{a}_{b}.png', nrow=11)
        # 遍历tensor中的图像，并在matplotlib中显示  
        # fig, axs = plt.subplots(16, 11, figsize=(20, 20))
        # for i in range(16):
        #     for j in range(11):
        #         # 使用transforms将tensor转换为PIL图像
        #         pil_image = t(fb[i, j].cpu())  # 如果在GPU上，需要使用.cpu()将tensor移回CPU
                
        #         # 显示图像在子图中
        #         axs[i, j].imshow(pil_image)
        #         axs[i, j].axis('off')  # 去除坐标轴

        # # 调整子图之间的间距
        # plt.subplots_adjust(wspace=0, hspace=0)

        # # 显示图像
        # plt.show()
        # fig = plt.figure()
        # for i in range(11):
        #     # 将numpy数组转换为图像对象。
        #     face_img_tensor = fb[1,i,:,:].float()
        #     _img = t(face_img_tensor)
        #     fig.add_subplot(4, 3,i+1)
        #     # vmin=0, vmax=255表示图像像素值的范围是0到255。
        #     plt.imshow(_img, cmap='gray', vmin=0, vmax=255)
        #     # plt.imshow(pmaps[:,:,i])
        # plt.show()

        # 使用torchvision.utils.save_image将张量的每个元素保存为图像
        # for i in range(16):
        #     # 从原始张量中选择第i个元素并添加一个维度，以将其形状变为[1, 11, 64, 64]
        #     image_tensor = fb[i]
        #     print(image_tensor.shape)
        #     # 保存图像
        #     torchvision.utils.save_image(image_tensor, f'tmp/fb_{i:03d}.png',nrow=11, padding=padding, normalize=False)
    print('done')

if __name__ == '__main__':
    main('folder')
