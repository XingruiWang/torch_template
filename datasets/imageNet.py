import os
import shutil
import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils.process_label import idx_map, idx_map_t, subclasses
# from utils.concept_dataset_label import idx_map, idx_map_t, subclasses

# https://arxiv.org/pdf/1802.07672.pdf 
# subclasses = {
#     'vehicles': ['n02701002','n02814533','n02930766','n03100240',
#                 'n03594945','n03769881','n03770679','n03930630',
#                 'n03977966','n04285008']
# }

# idx_map = {0: 407, 1: 436, 2: 468, 3: 511, 4: 609, 
#            5: 654, 6: 656, 7: 717, 8: 734, 9: 817}

# idx_map_t = {v:k for k, v in idx_map.items()}

def get_Imagenet_classes(subclass, use_ori_id = False, exclude = False):
    if not subclass:
        classes_name = os.listdir('/lab/tmpig23c/u/andy/ILSVRC/Annotations/CLS-LOC/train')
    elif exclude:
        classes_name = [i for i in os.listdir('/lab/tmpig23c/u/andy/ILSVRC/Annotations/CLS-LOC/train') if i not in subclasses[subclass]]
    else:
        classes_name = subclasses[subclass]
    classes_name = sorted(classes_name)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes_name)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes_name)}

    #######################
    use_ori_id = use_ori_id if not exclude else True
    if use_ori_id:
        all_classes_name = os.listdir('/lab/tmpig23c/u/andy/ILSVRC/Annotations/CLS-LOC/train')
        all_classes_name = sorted(all_classes_name)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(all_classes_name)}
        idx_to_class = {i: cls_name for i, cls_name in enumerate(all_classes_name)}
    #######################
    # for c in classes_name:
    #     print(c, class_to_idx[c])
    return classes_name, class_to_idx, idx_to_class


class ILSVRC(data.Dataset):
    def __init__(self, root, set_type='train', classes = None, transform = True, use_ori_id = False, exclude = False):
        self.root = root
        self.set_type = set_type

        self.set_transform(set_type, transform) 

        self.classes_name, self.class_to_idx, self.idx_to_class = get_Imagenet_classes(classes, use_ori_id, exclude)
        self.n_class = len(self.classes_name)

        self.img_dir = os.path.join(self.root, 'Data', 'CLS-LOC')
        self.anno_dir = os.path.join(self.root, 'Annotations', 'CLS-LOC')
        self.class_dir = os.path.join(self.root, 'ImageSets', 'CLS-LOC')

        self.load_file()

    def load_file(self, load_from_xml = False):

        set_type = self.set_type

        self.files = []
        self.classes = []

        if set_type == 'train':
            self.img_dir = os.path.join(self.img_dir, 'train')
            self.anno_dir = os.path.join(self.anno_dir, 'train')
            self.class_dir = os.path.join(self.class_dir, 'train_cls.txt')

            for c in tqdm(self.classes_name):
                for i in os.listdir(os.path.join(self.img_dir, c)):
                    img_path = os.path.join(self.img_dir, c, i)
                    self.files.append(img_path)
                    self.classes.append(c)

        elif set_type == 'val':
            if load_from_xml:
                print("Parsing val annotation...")
                self.img_dir = os.path.join(self.img_dir, 'val')
                self.anno_dir = os.path.join(self.anno_dir, 'val')
                self.class_dir = os.path.join(self.class_dir, 'val.txt')

                for i in tqdm(os.listdir(self.img_dir)):
                    anno_path = os.path.join(self.anno_dir, i.split('.')[0] + '.xml')
                    # parse an xml file by name
                    tree = ET.parse(anno_path)
                    root = tree.getroot()
                    c = root.find('object/name').text 
                    
                    if c in self.classes_name:             
                        self.classes.append(c)      
                        img_path = os.path.join(self.img_dir, i)
                        self.files.append(img_path)   

                        # make new validation 
                        # save_path = os.path.join(self.root, 'Data', 'CLS-LOC', 'validation', c)
                        # if not os.path.exists(save_path):
                        #     os.makedirs(save_path)
                        # shutil.copyfile(img_path, os.path.join(save_path, i))
            else:
                self.img_dir = os.path.join(self.img_dir, 'validation')
                self.anno_dir = os.path.join(self.anno_dir, 'val')
                self.class_dir = os.path.join(self.class_dir, 'val.txt')

                for c in tqdm(self.classes_name):
                    for i in os.listdir(os.path.join(self.img_dir, c)):
                        img_path = os.path.join(self.img_dir, c, i)
                        self.files.append(img_path)
                        self.classes.append(c)               

        elif set_type == 'test':
            self.img_dir = os.path.join(self.img_dir, 'test')
            self.anno_dir = os.path.join(self.anno_dir, 'test')
            self.class_dir = os.path.join(self.class_dir, 'test.txt')
  
        else:
            raise ValueError("Undefined dataset type: %s"%(set_type))

 
    def set_transform(self, set_type, transform):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        if set_type == 'train':
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_dir = self.files[idx]
        img = cv2.imread(img_dir)
        img = img[:, :, ::-1]
        img = self.transform(Image.fromarray(img))

        class_name = self.classes[idx]
        label = self.class_to_idx[class_name]

        return img, label

