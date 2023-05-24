'''
convert pseudo_label.json to image folder format
name format = "camid_name.jpg"
'''
import json
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type= str, default= "")
    parser.add_argument('--output_dir', type= str, default= "")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    args = get_args_parser()

    path = args.json_path
    output_dir = Path(args.output_dir)

    with open(path, "r") as f:
        data = json.load(f)

    for pid, items in tqdm(list(data.items())):
        class_dir = output_dir.joinpath(str(pid))
        class_dir.mkdir(parents=True, exist_ok=True)
    
        for (image_path, cam_id) in items:
            image_path = Path(image_path)
            new_image_path = class_dir.joinpath(f"{cam_id}_{image_path.name}")
            shutil.copy(image_path, new_image_path)

    print("Doneee")