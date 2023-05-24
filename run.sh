# # pretrain - huấn luyện
# CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/pretrain.yml \
#   DATASETS.DIR "/mnt/nvme0n1/datasets/reid/research_data/" \
#   DATASETS.SOURCE "dukemtmc" \
#   DATASETS.TARGET "market1501" \
#   OUTPUT_DIR "log/duke2market_nam/pretrain" \
#   GPU_Device '[0]' \
#   MODE 'pretrain' \
#   MODEL.ARCH "resnet50"


# # train- phân cụm data tại DATASETS.TARGET
# CUDA_VISIBLE_DEVICES=0 python main.py --config-file "configs/mutualrefine.yml"\
#             DATASETS.DIR "/mnt/nvme0n1/phuongnam/secret-reid" \
#             DATASETS.SOURCE "dukemtmc"\
#             DATASETS.TARGET "data_raw/PMC_sup_20220411" \
#             CHECKPOING.PRETRAIN_PATH "pretrain/checkpoint_new_market78.pth.tar"\
#             OUTPUT_DIR "log/PMC_sup_20220411" \
#             GPU_Device "[0]"\
#             MODE "mutualrefine" \
#             MODEL.ARCH "resnet50"\
#             OPTIM.EPOCHS "50"\
#             OPTIM.LR "0.0000875"\
#             DATALOADER.BATCH_SIZE "16"\
#             DATALOADER.ITERS "5000"


# # crop rotate image cam fisheye
python test_anno.py --json_path "/mnt/nvme0n1/datasets/fisheye/annotations/20220609_images.json" \
                    --folder_image "/mnt/nvme0n1/datasets/fisheye/20220609/20220609_images/" \
                    --folder_save_rotate "data_raw/PMC_sup_20220609"
#                     #--show True



# # convert pseudo label
# python scripts/convert_pseudo_label.py \
#         --json_path "log/PMC_sup_20220411/pseudo_label.json" \
#         --output_dir "PMC_sup/PMC_sup_20220411"