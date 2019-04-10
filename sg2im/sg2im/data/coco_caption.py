import json, os, random, math, io
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import warnings
warnings.filterwarnings('ignore')

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from .utils import imagenet_preprocess, Resize

class CocoCaptionDataSet(Dataset):
  def __init__(self, image_dir,captions_json,instances_json, stuff_json=None,
               stuff_only=True, image_size=(64, 64), mask_size=16,
               normalize_images=True, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None):
    super(Dataset,self).__init__()

    if stuff_only and stuff_json is None:
        print('WARNING: Got stuff_only=True but stuff_json=None.')
        print('Falling back to stuff_only=False.')


    self.image_dir = image_dir
    self.mask_size = mask_size
    self.max_samples = max_samples
    self.normalize_images = normalize_images
    self.include_relationships = include_relationships
    self.set_image_size(image_size)
    self.tokenizer = nlp.data.NLTKMosesTokenizer()

    captions_data = None
    if captions_json is not None and captions_json != '':
        with open(captions_json,'r') as f:
            captions_data = json.load(f)

    with open(instances_json, 'r') as f:
        instances_data = json.load(f)

    stuff_data = None
    if stuff_json is not None and stuff_json != '':
        with open(stuff_json, 'r') as f:
            stuff_data = json.load(f)

    self.image_ids = []
    self.image_id_to_filename = {}
    self.image_id_to_size = {}
    for image_data in instances_data['images']:
        image_id = image_data['id']
        filename = image_data['file_name']
        width = image_data['width']
        height = image_data['height']
        self.image_ids.append(image_id)
        self.image_id_to_filename[image_id] = filename
        self.image_id_to_size[image_id] = (width, height)

    self.vocab = {
        'object_name_to_idx': {},
        #'pred_name_to_idx': {},
    }

    object_idx_to_name = {}
    all_instance_categories = []
    for category_data in instances_data['categories']:
        category_id = category_data['id']
        category_name = category_data['name']
        all_instance_categories.append(category_name)
        object_idx_to_name[category_id] = category_name
        self.vocab['object_name_to_idx'][category_name] = category_id
    all_stuff_categories = []
    if stuff_data:
        for category_data in stuff_data['categories']:
            category_name = category_data['name']
            category_id = category_data['id']
            all_stuff_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id

    if instance_whitelist is None:
        instance_whitelist = all_instance_categories
    if stuff_whitelist is None:
        stuff_whitelist = all_stuff_categories
    category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

    # Add object data from instances
    self.image_id_to_objects = defaultdict(list)
    for object_data in instances_data['annotations']:
        image_id = object_data['image_id']
        _, _, w, h = object_data['bbox']
        W, H = self.image_id_to_size[image_id]
        box_area = (w * h) / (W * H)
        box_ok = box_area > min_object_size
        object_name = object_idx_to_name[object_data['category_id']]
        category_ok = object_name in category_whitelist
        other_ok = object_name != 'other' or include_other
        if box_ok and category_ok and other_ok:
            self.image_id_to_objects[image_id].append(object_data)

    self.image_id_to_captions = defaultdict(list)
    # Add object data from stuff
    if captions_data:
        image_ids_with_caption = set()
        for caption_data in captions_data['annotations']:
            image_id = caption_data['image_id']
            image_ids_with_caption.add(image_id)
            caption = caption_data['caption']
            self.image_id_to_captions[image_id].append(caption)#More than one caption data each image
        #new_image_ids = []
        #for image_id in self.image_ids:
            #if image_id in image_ids_with_caption:
                #new_image_ids.append(image_id)
        #self.image_ids = new_image_ids # Make sure all image has captions

    # Add object data from stuff
    if stuff_data:
        image_ids_with_stuff = set()
        for object_data in stuff_data['annotations']:
            image_id = object_data['image_id']
            image_ids_with_stuff.add(image_id)
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)
        if stuff_only:
            new_image_ids = []
            for image_id in self.image_ids:
                if image_id in image_ids_with_stuff:
                    new_image_ids.append(image_id)
            self.image_ids = new_image_ids

            all_image_ids = set(self.image_id_to_filename.keys())
            image_ids_to_remove = all_image_ids - image_ids_with_stuff
            for image_id in image_ids_to_remove:
                self.image_id_to_filename.pop(image_id, None)
                self.image_id_to_size.pop(image_id, None)
                self.image_id_to_objects.pop(image_id, None)
                self.image_id_to_captions.pop(image_id,None)


    # COCO category labels start at 1, so use 0 for __image__
    self.vocab['object_name_to_idx']['__image__'] = 0

    # Build object_idx_to_name
    name_to_idx = self.vocab['object_name_to_idx']
    assert len(name_to_idx) == len(set(name_to_idx.values()))
    max_object_idx = max(name_to_idx.values())
    idx_to_name = ['NONE'] * (1 + max_object_idx)
    for name, idx in self.vocab['object_name_to_idx'].items():
        idx_to_name[idx] = name
    self.vocab['object_idx_to_name'] = idx_to_name

    # Prune images that have too few or too many objects
    new_image_ids = []
    total_objs = 0
    for image_id in self.image_ids:
        num_objs = len(self.image_id_to_objects[image_id])
        total_objs += num_objs
        if min_objects_per_image <= num_objs <= max_objects_per_image:
            new_image_ids.append(image_id)

    #Keep one caption for each img
    self.image_ids = new_image_ids
    new_img_to_captions = defaultdict(list)
    for image_id in self.image_ids:
        captions = [cap for cap in self.image_id_to_captions[image_id]]
        caption = random.choice(captions)
        new_img_to_captions[image_id].append(caption)
    self.image_id_to_captions = new_img_to_captions


  def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

  def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

  def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

  def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_id = self.image_ids[index]
        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)

        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = self.transform(image.convert('RGB'))

        captions = []
        for caption in self.image_id_to_captions[image_id]:
            caption = ['<bos>'] + self.tokenizer(caption.lower()) + ['<sos>']
            captions.append(caption)
        return image,captions

  def getCounterandVocab(self):
      captions = []
      for image_id in self.image_ids:
          for caption in self.image_id_to_captions[image_id]:
              caption = ['<bos>'] + self.tokenizer(caption.lower()) + ['<sos>']
              captions.append(caption)
      counter = nlp.data.count_tokens([word for sentence in captions for word in sentence])
      vocab = nlp.Vocab(counter)
      return counter,vocab



def coco_caption_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    """
    all_captions,all_imgs = [],[]
    #all_cap_to_img = []
    for i, (img,captions) in enumerate(batch):
        #C = len(captions)
        all_imgs.append(img[None])
        all_captions.append(captions)
        #all_cap_to_img.extend(np.full(C,i))
    out = (all_imgs,all_captions)
    return out