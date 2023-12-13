# CUDA_VISIBLE_DEVICES=0,1 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.11.yml > runlog/train_LapSRNV4.11_x8_600k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.13.yml > runlog/v4.13_allface_600k_bs32_V100_1109.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.13_11face_by4.10.yml > runlog/v4.13base4.10_11face_400k_bs32_V100_1121.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.13_oneface_by4.10.yml > runlog/v4.13base4.10_oneface_400k_bs32_V100_1124.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.13_oneface.yml > runlog/v4.13_oneface_600k_bs32_V100_1114.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8.yml > runlog/v4.10_noLPE_600k_bs64_V100_1119.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_BSRN_x4.yml > runlog/BSRN_x8_11.26.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8_FAN60k.yml > runlog/train_LapSRNV4.10_x8_FAN60k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 \
# nohup ./scripts/dist_train.sh 2 options/train/FSR/train_LapSRNV4.11.yml > runlog/train_LapSRNV4.11_x8_600k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,2 \
# nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 \
# basicsr/train.py -opt options/train/FSR/train_LapSRNV4.11_allface.yml --launcher pytorch > runlog/train_LapSRNV4.11_all_face_x8_600k_2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 \
# nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=5322 \
# basicsr/train.py -opt options/train/FSR/train_LapSRNV4.12.yml --launcher pytorch > runlog/train_LapSRNV4.12_x8_600k.log 2>&1 &