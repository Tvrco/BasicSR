from torchvision import transforms
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_folder,paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_three_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import numpy as np
import os
from PIL import Image
import cv2
@DATASET_REGISTRY.register()
class BFSRDataset(data.Dataset):

    def __init__(self, opt):
        super(BFSRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder64, self.lq_folder32, self.lq_folder16 = opt['dataroot_gt'], opt[
            'dataroot_lq64'], opt['dataroot_lq32'], opt['dataroot_lq16']
        self.faceb_gt = opt['faceb_gt']
        # print(self.faceb_gt)
        self.list_boundary = [
            "_full_boundary",
            "_left_eyebow",
            "_right_eyebow",
            "_nose_bridge",
            "_nose_round",
            "_left_eye",
            "_right_eye",
            "_upper_mouth",
            "_upper_mouth_down",
            "_lower_mouth_up",
            "_lower_mouth_down"]
        # self.input_transform = transforms.Compose([transforms.ToTensor()])
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
                self.faceb_path = paths_from_folder(self.faceb_gt)
            else:
                self.paths = paired_paths_from_folder([self.lq_folder16, self.gt_folder], ['lq', 'gt'],
                                                      self.filename_tmpl)



    def _get_boundarymaps(self, index,size,gauss_flag=False):
        boundarymaps = np.zeros((size, size, len(self.list_boundary)))
        boundarymaps_list = []
        try:
            img_index = self.paths[index]['gt_path'].split('/')[-1].split('.')[0]
        except:
            img_index = self.paths[index]['gt_path'].split('\\')[-1].split('.')[0]
        # print(f"img_index:{img_index}")
        for i, tail in enumerate(self.list_boundary):
            line_path = os.path.join(self.faceb_gt,str(img_index) + tail +".jpg") # 固定从64*64获取人脸先验，32*32,16*16直接resize
            # print("dir-------", line_path)
            if not os.path.exists(line_path):
                print(f'不存在该边界线文件:{line_path}')
                continue
            face_line = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
            # ------------------------------------------
            # 使用高斯核
            if gauss_flag == True:
                if size == 32:
                    face_line = cv2.GaussianBlur(face_line,(5,5),0)
                else:
                    face_line = cv2.GaussianBlur(face_line,(9,9),0)
            # ------------------------------------------
            if size != 128:
                face_line = cv2.resize(face_line, (size, size), interpolation=cv2.INTER_CUBIC)
            boundarymaps[:,:,i] = face_line # 64*64*11
            boundarymaps_list.append(line_path)
        boundarymaps = boundarymaps.astype(np.float32)/ 255.0
        # boundarymaps = self.input_transform(boundarymaps)
        return boundarymaps,boundarymaps_list


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
        fb_32,fb_path_list = self._get_boundarymaps(index,size=32,gauss_flag=False)
        fb_64,_ = self._get_boundarymaps(index,size=64,gauss_flag=False)
        fb_128,_ = self._get_boundarymaps(index,size=128,gauss_flag=False)

        # augmentation for training
        if self.opt['phase'] == 'train':
            # gt_size = self.opt['gt_size']
            lq32_path = self.paths[index]['lq32_path']
            img_bytes = self.file_client.get(lq32_path, 'lq32')
            # print(img_bytes)
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
            img_gt, img_lq, img_lq32, img_lq64 = img2tensor([img_gt, img_lq, img_lq32, img_lq64],bgr2rgb=True,float32=True)
            fb_32 = img2tensor(fb_32,bgr2rgb=False,float32=True)
            fb_64 = img2tensor(fb_64,bgr2rgb=False,float32=True)
            fb_128 = img2tensor(fb_128,bgr2rgb=False,float32=True)
        else:
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_lq32, self.mean, self.std, inplace=True)
            normalize(img_lq64, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        if self.opt['phase'] == 'train':
            # print('boundarymaps shape : ',boundarymaps.shape)
            # print(fb_path_list)
            return {
                'lq': img_lq,
                'lq32': img_lq32,
                'lq64': img_lq64,
                'gt': img_gt,
                'fb_32' : fb_32,
                'fb_64' : fb_64,
                'fb_128' : fb_128,
                'lq_path': lq_path,
                'lq32_path': lq32_path,
                'lq64_path': lq64_path,
                'gt_path': gt_path,
                'fb_path': fb_path_list  
            }
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class BFSRDataset_onefaceb(data.Dataset):

    def __init__(self, opt):
        super(BFSRDataset_onefaceb, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder64, self.lq_folder32, self.lq_folder16 = opt['dataroot_gt'], opt[
            'dataroot_lq64'], opt['dataroot_lq32'], opt['dataroot_lq16']
        self.faceb_gt = opt['faceb_gt']
        # print(self.faceb_gt)
        # self.input_transform = transforms.Compose([transforms.ToTensor()])
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.opt['phase'] == 'train':
            self.paths = paired_paths_from_three_folder(
                [self.lq_folder16, self.lq_folder32, self.lq_folder64, self.gt_folder],
                ['lq', 'lq32', 'lq64', 'gt'], self.filename_tmpl)
            self.faceb_path = paths_from_folder(self.faceb_gt)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder16, self.gt_folder], ['lq', 'gt'],
                                                    self.filename_tmpl)



    def _get_boundarymaps(self, index,size,gauss_flag=False):

        try:
            img_index = self.paths[index]['gt_path'].split('/')[-1].split('.')[0]
        except:
            img_index = self.paths[index]['gt_path'].split('\\')[-1].split('.')[0]
        # print(f"img_index:{img_index}")

        line_path = os.path.join(self.faceb_gt,str(img_index) +".jpg") # 固定从64*64获取人脸先验，32*32,16*16直接resize
        # print("dir-------", line_path)
        if not os.path.exists(line_path):
            print(f'不存在该边界线文件:{line_path}')
        face_line = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
        # ------------------------------------------
        # 使用高斯核
        if gauss_flag == True:
            if size == 32:
                face_line = cv2.GaussianBlur(face_line,(5,5),0)
            else:
                face_line = cv2.GaussianBlur(face_line,(9,9),0)
        # ------------------------------------------
        if size != 128:
            face_line = cv2.resize(face_line, (size, size), interpolation=cv2.INTER_CUBIC)
        # 将图像转换为单通道的灰度图像
        face_line = face_line[:, :, np.newaxis]
        face_line = face_line.astype(np.float32)/ 255.0
        # print(face_line.shape)
        # boundarymaps = self.input_transform(boundarymaps)
        return face_line,line_path


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
        fb_32,fb_path_list = self._get_boundarymaps(index,size=32,gauss_flag=False)
        fb_64,_ = self._get_boundarymaps(index,size=64,gauss_flag=False)
        fb_128,_ = self._get_boundarymaps(index,size=128,gauss_flag=False)

        # augmentation for training
        if self.opt['phase'] == 'train':
            # gt_size = self.opt['gt_size']
            lq32_path = self.paths[index]['lq32_path']
            img_bytes = self.file_client.get(lq32_path, 'lq32')
            # print(img_bytes)
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
            img_gt, img_lq, img_lq32, img_lq64 = img2tensor([img_gt, img_lq, img_lq32, img_lq64],bgr2rgb=True,float32=True)
            fb_32 = img2tensor(fb_32,bgr2rgb=False,float32=True)
            fb_64 = img2tensor(fb_64,bgr2rgb=False,float32=True)
            fb_128 = img2tensor(fb_128,bgr2rgb=False,float32=True)
        else:
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_lq32, self.mean, self.std, inplace=True)
            normalize(img_lq64, self.mean, self.std, inplace=True)
            normalize(fb_32, self.mean, self.std, inplace=True)
            normalize(fb_64, self.mean, self.std, inplace=True)
            normalize(fb_128, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        if self.opt['phase'] == 'train':
            # print('boundarymaps shape : ',boundarymaps.shape)
            # print(fb_path_list)
            return {
                'lq': img_lq,
                'lq32': img_lq32,
                'lq64': img_lq64,
                'gt': img_gt,
                'fb_32' : fb_32,
                'fb_64' : fb_64,
                'fb_128' : fb_128,
                'lq_path': lq_path,
                'lq32_path': lq32_path,
                'lq64_path': lq64_path,
                'gt_path': gt_path,
                'fb_path': fb_path_list  
            }
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
