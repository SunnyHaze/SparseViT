import cv2
import random
import numpy as np
# Augmentation library
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2


def get_albu_transforms(type_ = 'resize', outputsize = 512):
    
    assert type_ in ['resize'] , "type_ must be 'resize' "
    trans = None
    if type_ == 'resize':
        trans = albu.Compose([
            albu.Resize(512, 512),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albu.Crop(0, 0, outputsize, outputsize),
            ToTensorV2(transpose_mask=True)
        ])        
    return trans