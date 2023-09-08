CUDA_VISIBLE_DEVICES=0 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8_FAN_2step_10k.yml > runlog/train_LapSRNV4.10_x8_100k_2step.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8_FAN.yml > runlog/train_LapSRNV4.10_x8_FAN10k_0.5.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8_FAN60k.yml > runlog/train_LapSRNV4.10_x8_FAN60k.log 2>&1 &
