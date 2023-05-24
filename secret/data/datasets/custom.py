from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile
import logging
from imutils.paths import list_images

from .base_dataset import BaseImageDataset

class CustomDataset(BaseImageDataset):
    """
    only train data

    cam_id
        image.jpg
    """

    def __init__(self, root, verbose=True, **kwargs):
        super(CustomDataset, self).__init__()
        self.dataset_dir = root

        train = self._process_dir(root)
        query = []
        gallery = []

        if verbose:
            logger = logging.getLogger('UnReID')
            logger.info(f"=> {self.dataset_dir} loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

    def _process_dir(self, dir_path):
        img_paths = list(list_images(dir_path))

        dataset = []
        cam_id = {}
        for img_path in img_paths:
            pid = -1
            cam_name = osp.basename(osp.dirname(img_path))
            if cam_id.get(cam_name, None) is None:
                cam_id[cam_name] = len(cam_id)
            
            dataset.append((img_path, pid, cam_id[cam_name]))

        return dataset
