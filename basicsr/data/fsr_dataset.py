from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_three_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FSRDataset(data.Dataset):

    def __init__(self, opt):
        super(FSRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder64, self.lq_folder32, self.lq_folder16 = opt['dataroot_gt'], opt[
            'dataroot_lq64'], opt['dataroot_lq32'], opt['dataroot_lq16']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            if self.opt['phase'] == 'train':
                self.paths = paired_paths_from_three_folder(
                    [self.lq_folder16, self.lq_folder32, self.lq_folder64, self.gt_folder],
                    ['lq', 'lq32', 'lq64', 'gt'], self.filename_tmpl)
            else:
                self.paths = paired_paths_from_folder([self.lq_folder16, self.gt_folder], ['lq', 'gt'],
                                                      self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            lq32_path = self.paths[index]['lq32_path']
            img_bytes = self.file_client.get(lq32_path, 'lq32')
            img_lq32 = imfrombytes(img_bytes, float32=True)

            lq64_path = self.paths[index]['lq64_path']
            img_bytes = self.file_client.get(lq64_path, 'lq64')
            img_lq64 = imfrombytes(img_bytes, float32=True)
            # # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # # flip, rotation 翻转和选择
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
            img_lq32 = bgr2ycbcr(img_lq32, y_only=True)[..., None]
            img_lq64 = bgr2ycbcr(img_lq64, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
        if self.opt['phase'] == 'train':
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq, img_lq32, img_lq64 = img2tensor([img_gt, img_lq, img_lq32, img_lq64],
                                                            bgr2rgb=True,
                                                            float32=True)
        else:
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_lq32, self.mean, self.std, inplace=True)
            normalize(img_lq64, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        if self.opt['phase'] == 'train':
            return {
                'lq': img_lq,
                'lq32': img_lq32,
                'lq64': img_lq64,
                'gt': img_gt,
                'lq_path': lq_path,
                'lq32_path': lq32_path,
                'lq64_path': lq64_path,
                'gt_path': gt_path
            }
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
