# CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/mutualrefine.yml \
#   DATASETS.DIR "/mnt/nvme0n1/datasets/reid/research_data" \
#   DATASETS.SOURCE "dukemtmc" \
#   DATASETS.TARGET "market1501" \
#   CHECKPOING.PRETRAIN_PATH "log/duke2market_nam/pretrain/checkpoint_new.pth.tar" \
#   OUTPUT_DIR "log/duke2market_nam/mutualrefine" \
#   GPU_Device '[1]' OPTIM.EPOCHS 50 \
#   MODE 'mutualrefine' \
#   MODEL.ARCH "resnet50"

# CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/mutualrefine.yml \
#   DATASETS.DIR "/mnt/nvme0n1/datasets/reid/PMC" \
#   DATASETS.SOURCE "dukemtmc" \
#   DATASETS.TARGET "pmc_reid_dyno_raw" \
#   CHECKPOING.PRETRAIN_PATH "log/duke2market_nam/pretrain/checkpoint_new.pth.tar" \
#   OUTPUT_DIR "log/duke2market_nam_PMC/mutualrefine" \
#   GPU_Device '[0]' OPTIM.EPOCHS 50 \
#   MODE 'mutualrefine' \
#   MODEL.ARCH "resnet50"

CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/mutualrefine.yml \
  DATASETS.DIR "/mnt/nvme0n1/phuongnam/yolox/" \
  DATASETS.SOURCE "dukemtmc" \
  DATASETS.TARGET "20220721_images_split_rotate" \
  CHECKPOING.PRETRAIN_PATH "log/duke2market_nam/pretrain/checkpoint_new.pth.tar" \
  OUTPUT_DIR "log/duke2market_nam_20220721_images_split_rotate/mutualrefine" \
  GPU_Device '[0]' OPTIM.EPOCHS 50 \
  MODE 'mutualrefine' \
  MODEL.ARCH "resnet50"
