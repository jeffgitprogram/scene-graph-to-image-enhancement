import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random
import io

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')
import mxnet as mx
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp

from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco_caption import CocoCaptionDataSet,coco_caption_collate_fn
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager
COCO_DIR = os.path.expanduser('../datasets/coco')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)


# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_captions_json',
         default=os.path.join(COCO_DIR, 'annotations/captions_train2017.json'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_captions_json',
         default=os.path.join(COCO_DIR, 'annotations/captions_val2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)


# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)
def build_coco_caption_dsets(args):
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'captions_json': args.coco_train_captions_json,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
    'stuff_only': args.coco_stuff_only,
    'image_size': args.image_size,
    'mask_size': args.mask_size,
    'max_samples': args.num_train_samples,
    'min_object_size': args.min_object_size,
    'min_objects_per_image': args.min_objects_per_image,
    'instance_whitelist': args.instance_whitelist,
    'stuff_whitelist': args.stuff_whitelist,
    'include_other': args.coco_include_other,
    'include_relationships': args.include_relationships,
  }
  train_dset = CocoCaptionDataSet(**dset_kwargs)
  num_objs = train_dset.total_objects()
  num_imgs = len(train_dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  # dset_kwargs['image_dir'] = args.coco_val_image_dir
  # dset_kwargs['captions_json'] = args.coco_val_captions_json
  # dset_kwargs['instances_json'] = args.coco_val_instances_json
  # dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  # dset_kwargs['max_samples'] = args.num_val_samples
  # val_dset = CocoCaptionDataSet(**dset_kwargs)


  return train_dset


def build_caption_loaders(args,vocab):

    train_dset = build_coco_caption_dsets(args)
    train_dset.coco_numerize_captions(vocab)
    print("Captions has been numerized./n")
    collate_fn = coco_caption_collate_fn

    # loader_kwargs = {
    #     'batch_size': args.batch_size,
    #     'num_workers': args.loader_num_workers,
    #     'shuffle': True,
    #     'batchify_fn': collate_fn,
    # }
    #train_loader = nlp.data.ShardedDataLoader(train_dset, **loader_kwargs)
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset,**loader_kwargs)
    print("data has been loaded")
    #vocab, counter = train_dset.getCounterandVocab()
    #loader_kwargs['shuffle'] = args.shuffle_val
    #val_loader = nlp.data.dataloader(val_dset, **loader_kwargs)
    return train_loader#, val_loader
def main(args):
    print("start")
    print(args)
    num_gpus = 1
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
    lstm, vocab = nlp.model.get_model('standard_lstm_lm_1500',
                                      dataset_name='wikitext-2',
                                      pretrained=True,
                                      ctx=context[0])
    print(vocab)

    #train_dset = build_coco_caption_dsets(args)
    train_loader = build_caption_loaders(args,vocab)
    class CaptionEncoder(gluon.HybridBlock):
      """Network for sentiment analysis."""
      def __init__(self, prefix=None, params=None):
            super(CaptionEncoder, self).__init__(prefix=prefix, params=params)
            with self.name_scope():
                self.embedding = None # will set with lm embedding later
                self.encoder = None # will set with lm encoder later


      def hybrid_forward(self, F, data,hiddens): # pylint: disable=arguments-differ
            encoded,hiddens = self.encoder(self.embedding(data),hiddens)  # Shape(T, N, C)
            return encoded,hiddens

    new_model = CaptionEncoder()
    new_model.embedding = lstm.embedding
    new_model.encoder = lstm.encoder
    new_model.begin_state = lstm.begin_state
    new_model.hybridize()


    print(new_model)

    def get_features(data, valid_lengths):
        # length = data.shape[1]
        batch_size = data.shape[0]
        hidden_state = new_model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context[0])
        # mask = mx.nd.arange(length).expand_dims(0).broadcast_axes(axis=(0,), size=(batch_size,))
        # mask = mask < valid_lengths.expand_dims(1).astype('float32')
        print(data.shape)
        data = mx.nd.transpose(data)
        output, (hidden, cell) = new_model(data, hidden_state)
        #hidden = mx.nd.transpose(hidden, axes=(1, 0, 2))
        print(hidden.shape)
        return (output, hidden)

    for batch in train_loader:
        imgs,captions,lens= batch
        imgs = imgs.cuda()
        data = captions.as_in_context(context[0])
        length = lens.as_in_context(context[0])
        features, hiddens = get_features(data, length)
        hiddens = nd.concat(hiddens[0], hiddens[1], dim=1)
        hiddens = torch.from_numpy(hiddens.as_in_context(mx.cpu()).asnumpy()).cuda()
        print(hiddens.size())
        print(imgs.size())





if __name__ == '__main__':
  args = parser.parse_args()
  main(args)