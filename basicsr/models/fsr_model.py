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
from .base_model import BaseModel
# from .sr_model import SRModel

@MODEL_REGISTRY.register()
class FSRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(FSRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)
        print(f'model_to_device done')
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:

            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.opt['network_g'].get('fb_single_face'):
            print(f'-------face landmark one MSE')
            self.face_all = True

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

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

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
            # if train_opt['dataroot_lq32'] != None :
            #     lx2_pix = self.cri_pix(self.srx2, self.lq32)
            #     lx4_pix = self.cri_pix(self.srx4, self.lq64)
            #     lx8_pix = self.cri_pix(self.output, self.gt)
            #     l_pix_three = lx8_pix+lx2_pix+lx4_pix # todo:歧义
            # else:
            #     lx8_pix = self.cri_pix(self.output, self.gt)
            #     l_pix_three = lx8_pix
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

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.cri_mse:
                    self.srx2,self.srx4,self.output,self.fbsr32,self.fbsr64,self.fbsr128 = self.net_g_ema(self.lq)
                else:
                    self.srx2,self.srx4,self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.cri_mse:
                    self.srx2,self.srx4,self.output,self.fbsr32,self.fbsr64,self.fbsr128 = self.net_g(self.lq)
                else:
                     self.srx2,self.srx4,self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])

            # print(f"sr_img_before:{visuals['result'].shape}\nsr_img_after:{sr_img.shape}")
            # print(visuals['sr_fb_128'].shape)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    if self.cri_mse:
                        batch_size = int(visuals['sr_fb_128'].shape[0])

                        if self.face_all:
                            for size in [128, 64, 32]:
                                key = f'sr_fb_{size}'
                                save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'fb_{size}_{img_name}_{current_iter}.png')
                                facec_img = tensor2img([visuals[key]])
                                imwrite(facec_img, save_img_path)
                        else:
                            for size in [128, 64, 32]:
                                key = f'sr_fb_{size}'
                                reshaped_tensor = visuals[key].view(batch_size * 11, 1, size, size)
                                save_fb_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_fb{size}_{current_iter}.png')
                                os.makedirs(osp.join(self.opt['path']['visualization'], img_name), exist_ok=True)
                                torchvision.utils.save_image(reshaped_tensor, f'{save_fb_path}', nrow=11)

                    else:
                        # Handle other cases if needed
                        pass
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        # 存放fb图片
        if self.cri_mse:
            out_dict['sr_fb_128'] = self.fbsr128.detach().cpu()
            out_dict['sr_fb_64'] = self.fbsr64.detach().cpu()
            out_dict['sr_fb_32'] = self.fbsr32.detach().cpu()
            if out_dict['sr_fb_128'].shape[1] == 1:
                self.face_all = True
        return out_dict
 
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
