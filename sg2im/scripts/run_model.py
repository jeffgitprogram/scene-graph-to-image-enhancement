#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, json, os

from imageio import imwrite
import torch
import mxnet as mx
from mxnet import nd
import gluonnlp as nlp

from sg2im.model import Sg2ImModel
from sg2im.captionencoder import CaptionEncoder
from sg2im.data.utils import imagenet_deprocess_batch
import sg2im.vis as vis


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/context.json')

parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])


def main(args):
  if not os.path.isfile(args.checkpoint):
    print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
    print('Maybe you forgot to download pretraind models? Try running:')
    print('bash scripts/download_models.sh')
    return

  if not os.path.isdir(args.output_dir):
    print('Output directory "%s" does not exist; creating it' % args.output_dir)
    os.makedirs(args.output_dir)
  layout_dir_name = os.path.join(args.output_dir, 'layout')
  if not os.path.isdir(layout_dir_name)
    print('Layout output directory "%s" does not exist; creating it' % layout_dir_name)
    os.makedirs(layout_dir_name)


  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')

  # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.to(device)

  #Load the caption encoder
  # Load Caption Encoder
  lstm, sent_vocab = nlp.model.get_model('standard_lstm_lm_650',
                                                   dataset_name='wikitext-2',
                                                   pretrained=True,
                                                   ctx=mx.cpu())
  # print(self.sent_vocab)
  caption_encoder = CaptionEncoder()
  caption_encoder.embedding = lstm.embedding
  caption_encoder.encoder = lstm.encoder
  caption_encoder.begin_state = lstm.begin_state
  caption_encoder.hybridize()
  print(caption_encoder)

  # Load the scene graphs
  with open(args.scene_graphs_json, 'r') as f:
    scene_graphs = json.load(f)
  #Generate hiddens for captions
  caption_hiddens = generateCaptionHidden(caption_encoder, sent_vocab,scene_graphs)
  # Run the model forward
  with torch.no_grad():
    imgs, boxes_pred, masks_pred, _, _, objs = model.forward_json(scene_graphs,caption_hiddens)
  imgs = imagenet_deprocess_batch(imgs)

  # Save the generated images
  for i in range(imgs.shape[0]):
    img_np = imgs[i].numpy().transpose(1, 2, 0)
    img_path = os.path.join(args.output_dir, 'img%06d.png' % i)
    imwrite(img_path, img_np)
    layout_plt = vis.draw_layout(model.vocab, objs[i], boxes_pred[i], masks_pred[i], show_boxes=True)
    layout_path = os.path.join(layout_dir_name, 'img%06dlayout.png' % i)
    layout_plt.savefig(layout_path)
    layout_plt.clf()


  # Draw the scene graphs
  if args.draw_scene_graphs == 1:
    for i, sg in enumerate(scene_graphs):
      sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
      sg_img_path = os.path.join(args.output_dir, 'sg%06d.png' % i)
      imwrite(sg_img_path, sg_img)

def generateCaptionHidden(encoder, vocab, scene_graphs):
  if isinstance(scene_graphs, dict):
  # We just got a single scene graph, so promote it to a list
    scene_graphs = [scene_graphs]
  tokenizer = nlp.data.NLTKMosesTokenizer()
  hidden_set = []
  for i, sg in enumerate(scene_graphs):
    sentence = tokenizer(sg['caption'].lower()) + ['<eos>']
    print(sentence)
    length = len(sentence)
    sentence = vocab[sentence]
    sentence = nd.array([sentence])
    length = nd.array(length)
    _, hiddens = get_features(encoder, sentence, length)
    hiddens = hiddens[1]
    hiddens = torch.from_numpy(hiddens.asnumpy())
    hidden_set.append(hiddens)
  hidden_set = torch.cat(hidden_set)
  hidden_set = hidden_set.cuda() if args.device == 'gpu' else hidden_set.cpu()
  return hidden_set


def get_features(self, model, data, valid_lengths):
    # length = data.shape[1]
    batch_size = data.shape[0]
    hidden_state = model.begin_state(func=mx.nd.zeros, batch_size=batch_size)
    # mask = mx.nd.arange(length).expand_dims(0).broadcast_axes(axis=(0,), size=(batch_size,))
    # mask = mask < valid_lengths.expand_dims(1).astype('float32')
    # print(data.shape)
    data = mx.nd.transpose(data)
    output, (hidden, cell) = model(data, hidden_state)
    # hidden = mx.nd.transpose(hidden, axes=(1, 0, 2))
    # print(hidden.shape)
    return (output, hidden)
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

