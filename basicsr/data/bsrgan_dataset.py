import os
import random

from torch.utils import data as data
from torchvision.transforms.functional import normalize
# from torchvision.transforms.v2.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder,paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor,degradation_bsrgan,degradation_bsrgan_plus
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DatasetBlindSR(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(DatasetBlindSR, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        '''
        # -----------------------------------------
        # dataset for BSRGAN
        # -----------------------------------------
        '''
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            self.paths = paths_from_folder(self.gt_folder)
        # print(self.paths)
    def __getitem__(self, index):
        print(1123222222222222222222222222222222222222222)
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        # lq_path = self.paths[index]['lq_path']
        lq_path = None
        # img_bytes = self.file_client.get(lq_path, 'lq')
        # img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            
            gt_size = self.opt['gt_size']
            '''
            # -----------------------------------------
            # BSRGAN 随机crop_patch_size
            # -----------------------------------------
            '''
            H, W, C = img_gt.shape
            # 随机crop_patch_size
            rnd_h_H = random.randint(0, max(0, H - gt_size))
            rnd_w_H = random.randint(0, max(0, W - gt_size))
            img_gt = img_gt[rnd_h_H:rnd_h_H + gt_size, rnd_w_H:rnd_w_H + gt_size, :]
            print(img_gt.shape)
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            if self.degradation_type == 'bsrgan':
                img_lq, img_gt = degradation_bsrgan(img_gt, scale, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_lq, img_gt = degradation_bsrgan_plus(img_gt, scale, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            if self.degradation_type == 'bsrgan':
                img_lq, img_gt = degradation_bsrgan(img_gt, scale, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_lq, img_gt = degradation_bsrgan_plus(img_gt, scale, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)
            # img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)
        if lq_path is None:
            lq_path = gt_path
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
