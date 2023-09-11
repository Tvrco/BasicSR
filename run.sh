# CUDA_VISIBLE_DEVICES=0,1 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.11.yml > runlog/train_LapSRNV4.11_x8_600k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8_FAN.yml > runlog/train_LapSRNV4.10_x8_FAN10k_0.5.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8_FAN60k.yml > runlog/train_LapSRNV4.10_x8_FAN60k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 \
# nohup ./scripts/dist_train.sh 2 options/train/FSR/train_LapSRNV4.11.yml > runlog/train_LapSRNV4.11_x8_600k.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 \
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 \
basicsr/train.py -opt options/train/FSR/train_LapSRNV4.11.yml --launcher pytorch > runlog/train_LapSRNV4.11_x8_600k.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 \
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 \
basicsr/train.py -opt options/train/FSR/train_LapSRNV4.12.yml --launcher pytorch > runlog/train_LapSRNV4.12_x8_600k.log 2>&1 &