from sympy import true
import torch
from collections import OrderedDict
from os import path as osp
import torchvision
import os
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .fsr_model import FSRModel
# from .sr_model import SRModel

@MODEL_REGISTRY.register()
class FSRGANModel(FSRModel):
    """Base SR model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)

        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('fan_opt'):
            self.cri_fan = build_loss(train_opt['fan_opt']).to(self.device)
        else:
            self.cri_fan = None

        if train_opt.get('mse_opt'):
            self.cri_mse = build_loss(train_opt['mse_opt']).to(self.device)
        else:
            self.cri_mse = None
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        logger = get_root_logger()
        if self.opt['network_g'].get('fb_single_face'):
            logger.info(f'-------face landmark one MSE by 1')
            self.face_all = True #代表使用全脸1张图
        else:
            if self.cri_mse:
                logger.info(f'-------face landmark one MSE by 11')
            else:
                logger.info(f'-------No face landmark')
            self.face_all = False #使用11张子图
        logger.info(f'-------Use GAN')
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        if 'lq32' in data:
            self.lq = data['lq'].to(self.device)
            self.lq32 = data['lq32'].to(self.device)
            self.lq64 = data['lq64'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            train_flage = true
            if 'fb_128' in data:
                self.fb_128 = data['fb_128'].to(self.device)
                self.fb_32 = data['fb_32'].to(self.device)
                self.fb_64 = data['fb_64'].to(self.device)
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
    def optimize_parameters(self, current_iter):
        # train_opt = self.opt['train']
        # print(train_opt)
        self.optimizer_g.zero_grad()
        if self.cri_mse:
            self.srx2,self.srx4,self.output,self.fbsr32,self.fbsr64,self.fbsr128 = self.net_g(self.lq)
        else:
            self.srx2,self.srx4,self.output = self.net_g(self.lq)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            lx2_pix = self.cri_pix(self.srx2, self.lq32)
            lx4_pix = self.cri_pix(self.srx4, self.lq64)
            lx8_pix = self.cri_pix(self.output, self.gt)
            if self.cri_fan:
                fan_pix_64 = self.cri_fan(self.srx4, self.lq64)
                fan_pix_128 = self.cri_fan(self.output, self.gt)
                fan_pix = fan_pix_64 + fan_pix_128
                l_pix_three = lx8_pix+lx2_pix+lx4_pix+fan_pix # todo:歧义
                loss_dict['l_fan'] = fan_pix
            elif self.cri_mse:
                mse_32 = self.cri_mse(self.fbsr32,self.fb_32)
                mse_64 = self.cri_mse(self.fbsr64,self.fb_64)
                mse_128 = self.cri_mse(self.fbsr128,self.fb_128)
                mse_pix = mse_32+mse_64+mse_128
                l_pix_three = lx8_pix+lx2_pix+lx4_pix+mse_pix # todo:歧义
                loss_dict['l_mse'] = mse_pix
            else:
                l_pix_three = lx8_pix+lx2_pix+lx4_pix# todo:歧义
            l_total += l_pix_three
            loss_dict['l_pix'] = l_pix_three
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # gan loss
        fake_g_pred = self.net_d(self.output)
        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
        l_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan

        l_total.backward()
        self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
