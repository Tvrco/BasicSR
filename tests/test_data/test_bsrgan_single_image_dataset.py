import yaml

from basicsr.data.single_image_dataset import SingleImageDataset
from basicsr.data.bsrgan_dataset import DatasetBlindSR


def test_DatasetBlindSR():
    """Test dataset: DatasetBlindSR"""

    opt_str = r"""
name: Test
type: DatasetBlindSR
dataroot_gt: E:\\PyProject\\data\\2K_Resolution\\DIV2K\\GTmod4
dataroot_lq: ~
io_backend:
    type: disk

shuffle_prob : 0.1
use_sharp : False
degradation_type : bsrgan
lq_patchsize : 64

gt_size : 128
use_hflip : True
use_rot : True

num_worker_per_gpu : 2
batch_size_per_gpu : 16
scale : 4

dataset_enlarge_ratio : 1
phase : train
mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
"""
    opt = yaml.safe_load(opt_str)

    dataset = DatasetBlindSR(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    # assert len(dataset) == 2  # whether to read correct meta info
    # print(len(dataset))  # whether to read correct meta info
    assert dataset.mean == [0.5, 0.5, 0.5]

    # test __getitem__
    result = dataset.__getitem__(0)
    print(result.keys())
    # check returned keys
    # expected_keys = ['lq', 'lq_path']
    # assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 128, 128)
    assert result['gt_path'] == 'E:\\\\PyProject\\\\data\\\\2K_Resolution\\\\DIV2K\\\\GTmod4\\0801.png'

    # # ------------------ test scan folder mode -------------------- #
    # opt.pop('meta_info_file')
    # opt['io_backend'] = dict(type='disk')
    # dataset = SingleImageDataset(opt)
    # assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    # assert len(dataset) == 2  # whether to correctly scan folders

    # # ------------------ test lmdb backend and with y channel-------------------- #
    # opt['dataroot_lq'] = 'tests/data/lq.lmdb'
    # opt['io_backend'] = dict(type='lmdb')
    # opt['color'] = 'y'
    # opt['mean'] = [0.5]
    # opt['std'] = [0.5]

    # dataset = DatasetBlindSR(opt)
    # assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    # assert len(dataset) == 2  # whether to read correct meta info
    # assert dataset.std == [0.5]

    # # test __getitem__
    # result = dataset.__getitem__(1)
    # # check returned keys
    # expected_keys = ['lq', 'lq_path']
    # assert set(expected_keys).issubset(set(result.keys()))
    # # check shape and contents
    # assert result['lq'].shape == (1, 90, 60)
    # assert result['lq_path'] == 'comic'


if __name__ == "__main__":
    test_DatasetBlindSR()