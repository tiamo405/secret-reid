CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/pretrain.yml \
  DATASETS.DIR "/mnt/nvme0n1/datasets/reid/research_data/" \
  DATASETS.SOURCE "dukemtmc" \
  DATASETS.TARGET "market1501" \
  OUTPUT_DIR "log/duke2market_nam/pretrain" \
  GPU_Device '[0]' \
  MODE 'pretrain' \
  MODEL.ARCH "resnet50"

