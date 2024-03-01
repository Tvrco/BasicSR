# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_LapSRNV4.10_x8_FAN60k.yml > runlog/train_LapSRNV4.10_x8_FAN60k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 \
# nohup ./scripts/dist_train.sh 2 options/train/FSR/train_LapSRNV4.11.yml > runlog/train_LapSRNV4.11_x8_600k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,2 \
# nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 \
# basicsr/train.py -opt options/train/FSR/train_LapSRNV4.11_allface.yml --launcher pytorch > runlog/train_LapSRNV4.11_all_face_x8_600k_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 \
# nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=5322 \
# basicsr/train.py -opt options/train/FSR/train_LapSRNV4.12.yml --launcher pytorch > runlog/train_LapSRNV4.12_x8_600k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/FSR/train_BSRFSR-GAN_inference_byori.yml > runlog/BSRFSR_GAN-ori-11.13.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u basicsr/train.py -opt options/train/FSR/train_BSRFSR-GAN_inference_bygan.yml > runlog/BSRFSR_GAN-gan-11.13.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C62.yml > runlog/CEBSDN_Helenx8_C62_2.27.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C58.yml > runlog/CEBSDN_Helenx8_C58_2.27.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C48.yml > runlog/CEBSDN_Helenx8_C48_2.27.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C64.yml > runlog/CEBSDN_Helenx8_C64_2.27.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C64.yml > runlog/CEBSDN_Helenx8_C64BS64_L1_100k_2.28.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C48.yml > runlog/CEBSDN_Helenx8_C48BS64_L1_100k_2.28.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C64_wo_dataen.yml > runlog/CEBSDN_wo_dataen_Helenx8_C64BS64_L1_100k_2.28.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C64_test.yml > runlog/CEBSDN_test_Helenx8_C64BS128_L1_300k_2.27.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C64_test2.yml > runlog/CEBSDN_test2_Helenx8_C64BS64_L1_600k.log 2>&1 &
#2.28
# CUDA_VISIBLE_DEVICES=0 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_BSRN_x8.yml > runlog/BSRN_Helen_x8_C64B8_L1_1000k_2.27.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C48_fine.yml > runlog/CEBSDN_fine_Helenx8_C48BS128_L1_300k_2.28.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C48_wo_Cat.yml> runlog/CEBSDN_wo_Cat_Helenx8_C48BS64_L1_100k_2.28.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u basicsr/train.py -opt options/train/CEBSDN/train_CEBSDN_Helen_C48_wo_CA.yml> runlog/CEBSDN_wo_CA_Helenx8_C48BS64_L1_100k_2.28.log 2>&1 &
#2.28_11
CUDA_VISIBLE_DEVICES=1 nohup python -u basicsr/train.py -opt options/train/CEBSDN/CelebA/train_CEBSDN_CelebA_C48.yml > runlog/CEBSDN_CelebAx8_C48BS64_L1_600k_2.28.log 2>&1 &