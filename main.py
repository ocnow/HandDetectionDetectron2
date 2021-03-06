# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ect1RZxJiFvLKsH1Sihwt3ZidQ8i5HTU
"""

# install dependencies: 
!pip install pyyaml==5.1
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab

# install detectron2: (Colab has CUDA 10.1 + torch 1.7)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
import torch
assert torch.__version__.startswith("1.7")
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

import torch, torchvision
import torch
assert torch.__version__.startswith("1.7")

#import torch
#assert torch.__version__.startswith("1.7")
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import pandas as pd
import scipy.io as sio
import os
from matplotlib import pyplot as plt

!wget http://vision.soic.indiana.edu/egohands_files/egohands_data.zip
!unzip egohands_data.zip > /dev/null

BASE_PATH = '_LABELLED_SAMPLES/'

folders = sorted(os.listdir('_LABELLED_SAMPLES'))

folder_names = []
image_names = []
polygons_master = []

for folder in folders:
    img_names = sorted([x for x in os.listdir(BASE_PATH + folder) if x.split('.')[-1] == 'jpg'])
    poly_file = sio.loadmat(BASE_PATH + folder +'/'+ 'polygons.mat')['polygons'][0]
    i = 0
    for img_name in img_names:
        folder_names.append(folder)
        image_names.append(img_name)
        polygons_master.append(poly_file[i])
        i = i + 1

df = pd.DataFrame({'folder_name':folder_names,'image_names':image_names,'polygons':polygons_master})

from sklearn.model_selection import train_test_split

train,test = train_test_split(df,test_size=0.2)

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
  df = None
  if img_dir == 'train':
    df = train
  else:
    df = test
  
  dataset_dicts = []
  i = 0
  for index, row in df.iterrows():
    i = i + 1
    #print("doing for index "+str(i))
    record = {}

    filename = os.path.join(BASE_PATH + row['folder_name'], row['image_names'])
    #height, width = cv2.imread(filename).shape[:2]

    record["file_name"] = filename
    record["image_id"] = row['image_names'].split('.')[0]
    record["height"] = 720
    record["width"] = 1280

    annos = row['polygons']
    objs = []
    
    for hand in annos:
      x_min = 2000
      y_min = 2000
      x_max = -1
      y_max = -1

      #array to hold x and y values of hand pixels
      poly = []

      if hand.size > 0:
        for pixel in hand:
          x = int(pixel[0])
          y = int(pixel[1])
          
          poly.append(x)
          poly.append(y)
          
          if  x > x_max:
              x_max = x
          if x < x_min:
              x_min = x
          if y > y_max:
              y_max = y
          if y < y_min:
              y_min = y
          
        #array to corners of bbox
        corners = [x_min,y_min,x_max,y_max]

        obj = {
              "bbox": corners,
              "bbox_mode": BoxMode.XYXY_ABS,
              "segmentation": [poly],
              "category_id": 0,
          }
        objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("hand_" + d, lambda d=d: get_balloon_dicts(d))
    MetadataCatalog.get("hand_" + d).set(thing_classes=["hand"])
balloon_metadata = MetadataCatalog.get("hand_train")

dataset_dicts = get_balloon_dicts("train")

d = dataset_dicts[150]
img = cv2.imread(d["file_name"])
visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
out = visualizer.draw_dataset_dict(d)
cv2_imshow(out.get_image()[:, :, ::-1])

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("hand_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 600   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts("test")

for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

