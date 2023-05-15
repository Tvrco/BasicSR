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
    opt['type'] = 'FSRDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = 'E:\\PyProject\\data\\classical_SR_datasets\\CelebA-HQ_ParsingMap\\test\\GTmod128'
        opt['dataroot_lq16'] = 'E:\\PyProject\\data\\classical_SR_datasets\\CelebA-HQ_ParsingMap\\test\\LRbicx8'
        opt['dataroot_lq32'] = 'E:\\PyProject\\data\\classical_SR_datasets\\CelebA-HQ_ParsingMap\\test\\LRbicx4'
        opt['dataroot_lq64'] = 'E:\\PyProject\\data\\classical_SR_datasets\\CelebA-HQ_ParsingMap\\test\\LRbicx2'
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
        if i > 1:
            break
        print(i)

        lq = data['lq']
        lq32 = data['lq32']
        lq64 = data['lq64']
        gt = data['gt']
        lq_path = data['lq_path']
        lq32_path = data['lq32_path']
        lq64_path = data['lq64_path']
        gt_path = data['gt_path']
        print(lq_path,'\n',lq32_path,'\n',lq64_path,'\n', gt_path)
        torchvision.utils.save_image(lq, f'tmp/lq_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(lq32, f'tmp/lq32_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(lq64, f'tmp/lq64_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    main('folder')
