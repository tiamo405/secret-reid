import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import math
import cv2
import imutils
import random
import os
import copy
import argparse
import cv2
import math
import numpy as np

def rotate_img(image, box):
    _, x, y, w, h, rad = box
    rad = rad / math.pi * 180
    
    height, width =image.shape[:2]

    # new_width = int(abs(math.cos(rad)) * width + abs(math.sin(rad)) * height)
    # new_height = int(abs(math.sin(rad)) * width + abs(math.cos(rad)) * height)

    M = cv2.getRotationMatrix2D((width / 2, height / 2), rad, 1)
    # M[0, 2] += (new_width - width) / 2
    # M[1, 2] += (new_height - height) / 2

    rotated_image = cv2.warpAffine(image, M, (width, height))
    cv2.imwrite("rotate.jpg", rotated_image)
    return rotated_image

def rotate_point(x, y, rad, image):
    cos_rad = math.cos(rad)
    sin_rad = math.sin(rad)
    rotation_matrix = np.array([[cos_rad, -sin_rad],
                                [sin_rad, cos_rad]])
    image_height,image_width = image.shape[:2]
    x = x - image_width /2
    y = image_height /2 - y
    point = np.array([[x], [y]])
    rotated_point = rotation_matrix.dot(point)
    rotated_x, rotated_y = np.squeeze(rotated_point)
    new_x = rotated_x + image_width /2
    new_y = image_height/2 - rotated_y

    return int(new_x), int(new_y)

def crop_img(image, h,w,x,y, scale = 1):
    w = int(w*scale)
    h = int(h* scale)

    crop_image = image[int(y - h/2): int(y + h/2), int(x - w/2): int(x + w/2)]
    try:
        cv2.imwrite('crop.jpg',crop_image)
    except:
        return image
    return crop_image

def npxywha2vertex(box):
    """
    use radian
    X=x*cos(a)-y*sin(a)
    Y=x*sin(a)+y*cos(a)
    """
    batch = box.shape[0]

    center = box[:,:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate two vector
    verti = np.empty((batch,2), dtype=np.float32)
    verti[:,0] = (h/2) * np.sin(rad)
    verti[:,1] = - (h/2) * np.cos(rad)

    hori = np.empty((batch,2), dtype=np.float32)
    hori[:,0] = (w/2) * np.cos(rad)
    hori[:,1] = (w/2) * np.sin(rad)

    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori
    
    tc = center + verti
    bc = center - verti

    return np.concatenate([tl,tr,br,bl], axis=1), np.concatenate([tc, bc], axis=1).reshape(2, 2).astype(np.int32).tolist()


def main(args):
    json_path = args.json_path
    folder_save_rotate = args.folder_save_rotate
    folder_image = args.folder_image

    if not os.path.exists(folder_save_rotate):
        os.makedirs(folder_save_rotate, exist_ok= True)


    data = {}
    bbox_rotate = {}
    json_path = Path(json_path)
    # if "merge" not in str(json_path):
    #     data_dir = json_path.parents[0].joinpath(name_folder_image)
    # else: 
    #     data_dir = json_path.parents[1]
    data_dir = Path(folder_image)
    with open(str(json_path), 'r') as f:
        json_data = json.load(f)

    path_id = {d['id']:d['file_name'] for d in json_data['images']}
    desc = f"{json_path.stem} - Scanning '{data_dir}' images and labels..."
    for ann in tqdm(json_data['annotations'], desc=desc):
        image_path = data_dir.joinpath(path_id[ann['image_id']])
        assert image_path.is_file(), f"File not found: {str(image_path)}"

        l = data.get(str(image_path), [np.empty((0, 9)), None, []])
        l_box = bbox_rotate.get(str(image_path), [np.empty((0,6)), None, []])
        if l[1] is None:
            img = Image.open(image_path)
            shape = img.size
            l[1] = shape
        shape = l[1]
        bbox = ann['bbox']
        # bbox[0] /= shape[0]
        # bbox[1] /= shape[1]
        # bbox[2] /= shape[0]
        # bbox[3] /= shape[1]
        
        bbox[4] = bbox[4] / 180 * math.pi

        w, h = bbox[2:4]
        if w > h:
            bbox[2], bbox[3] = h, w
        
        vector_1 = [0, -10]
        vector_2 = [bbox[0]-shape[0]/2, bbox[1]-shape[1]/2]
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        if vector_2[0] < 0: angle = -angle
        bbox[4] = angle

        bbox = np.array(bbox)
        bbox, (tc, bc) = npxywha2vertex(bbox[np.newaxis])

        bbox = np.concatenate([np.zeros((1, 1)), bbox], axis=1)
        
        l[2].append((tc, bc))
        l[0] = np.concatenate([l[0], np.array(bbox)])
        data[str(image_path)] = l

        box = ann['bbox']
        box[4] = angle
        box = np.array(box)
        box = np.expand_dims(box, axis=0)
        box = np.concatenate([np.zeros((1, 1)), box], axis=1)
        l_box[0] = np.concatenate([l_box[0], np.array(box)])
        bbox_rotate[str(image_path)] = l_box

    image_paths = list(data.keys())
    random.shuffle(image_paths)
    # image_paths.sort(reverse=True)
    for image_path in image_paths:
        print(image_path)
        image = cv2.imread(image_path)
        image_copy = copy.deepcopy(image)
        det = data[image_path]
        bbox = bbox_rotate[image_path]

        cv2.circle(image, (int(image.shape[1]/2), int(image.shape[0]/2)), 5, (0, 0, 255), -1)
        for i, (conf, *xyxyxyxy) in enumerate(det[0]):
            xyxyxyxy = [int(i) for i in xyxyxyxy]
            pts = np.array([xyxyxyxy[i:i+2] for i in range(0, len(xyxyxyxy), 2)], np.int32)

            pts = pts.reshape((-1, 1, 2))
            # cv2.rectangle(image, (x1, y1), (x3, y3), (0, 0, 255), 3)
            cv2.putText(image, f'{conf:.2f}', (xyxyxyxy[0],xyxyxyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            image = cv2.polylines(image, [pts], True, (0, 0, 255), 2)
            cv2.imwrite('debug.jpg', image)

            box = bbox[0][i]
            _, x, y ,w, h, rad = box
            img = rotate_img(image_copy, box)
            new_x, new_y = rotate_point(x,y, rad, img)
            rotated_image = crop_img(img, h,w, new_x, new_y)
            file_name = os.path.basename(image_path)
            name_rotated_image =file_name.split('.')[0]+ '_' + str(i).zfill(4) + '.jpg'
            cv2.imwrite(os.path.join(folder_save_rotate, name_rotated_image), rotated_image)

            print(name_rotated_image)
        if args.show:
            cv2.imshow("image", imutils.resize(image, width=1000))
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type= str, default= "")
    parser.add_argument('--folder_image', type= str, default= "")
    parser.add_argument('--folder_save_rotate', type= str, default= "")
    parser.add_argument("--show", type= bool, default= False)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = get_args_parser()
    main(args= args)
