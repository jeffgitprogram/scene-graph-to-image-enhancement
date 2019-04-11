import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random
import io

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import mxnet as mx
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp


print("start")
num_gpus = 1
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
elmo_intro = """
Extensive experiments demonstrate that ELMo representations work extremely well in practice.
We first show that they can be easily added to existing models for six diverse and challenging language understanding problems, including textual entailment, question answering and sentiment analysis.
The addition of ELMo representations alone significantly improves the state of the art in every case, including up to 20% relative error reductions.
For tasks where direct comparisons are possible, ELMo outperforms CoVe (McCann et al., 2017), which computes contextualized representations using a neural machine translation encoder.
Finally, an analysis of both ELMo and CoVe reveals that deep representations outperform those derived from just the top layer of an LSTM.
Our trained models and code are publicly available, and we expect that ELMo will provide similar gains for many other NLP problems.
"""
list = [[9, 1473, 5, 15266, 0, 7871, 85, 7, 2, 295, 4, 1], [9, 400, 18, 9, 0, 2380, 9, 4348, 4, 1], [9, 462, 14397, 10801, 7, 2, 4792, 5, 9, 19616, 4, 1], [52, 0, 0, 307, 20, 999, 78, 0, 6, 999, 78, 0, 307, 8, 97, 4, 1], [9, 511, 1898, 510, 6, 9828, 8417, 6, 15321, 1], [9, 400, 7, 1016, 17711, 6958, 9, 4010, 20, 9, 9418, 4, 1], [81, 8146, 8124, 10208, 78, 307, 8, 175, 64, 18, 9, 5132, 4, 1], [0, 4817, 9, 1769, 15, 24, 888, 8, 0, 71, 1], [9, 0, 3, 0, 578, 400, 3087, 10677, 6, 9, 3458, 3694, 0, 63, 2, 3385, 4, 1], [52, 26561, 36, 907, 109, 9, 17986, 440, 4, 1], [9, 4988, 0, 770, 118, 9, 2177, 7, 9, 440, 4, 1], [9, 23116, 20, 9, 3109, 2911, 6, 9, 7667, 7, 9, 19616, 1], [9, 0, 20381, 20, 9, 0, 1316, 10801, 18, 45, 0, 4, 1], [3755, 0, 197, 1898, 427, 2, 295, 7, 9, 0, 2187, 4, 1], [2, 963, 25, 8514, 52, 2778, 0, 1898, 85, 1721, 2, 2778, 6, 614, 2, 295, 4, 1], [9, 0, 24, 0, 9, 23233, 2030, 7, 2, 17725, 1], [125, 0, 6119, 68, 168, 20, 64, 2538, 180, 1], [9, 20417, 400, 14181, 18, 26, 6587, 18, 9, 5937, 4, 1], [9, 950, 6014, 47, 462, 3755, 18, 9, 2354, 4, 1], [9, 2022, 6, 510, 0, 5982, 18, 2, 2778, 4, 1], [52, 0, 3, 49, 898, 3, 49, 1488, 307, 8, 175, 64, 4, 1], [9, 26866, 5, 11387, 0, 18, 9, 462, 5992, 18, 9, 2867, 3360, 4, 1], [9, 6823, 2187, 14323, 1624, 2, 3302, 18, 9, 2565, 1], [2, 243, 5, 0, 24, 11316, 7, 2, 4145, 4, 1], [9, 400, 24, 592, 6674, 20, 26, 32402, 7, 26, 892, 4, 1], [9, 185, 5967, 6602, 5034, 197, 9, 18135, 2354, 4, 1], [9, 400, 18, 0, 0, 828, 197, 9, 1899, 4, 1], [9, 2654, 20, 9, 16293, 5, 2654, 25408, 7871, 25, 29, 4, 1], [9, 27225, 16417, 2, 3574, 15, 24, 18, 2, 531, 4, 1], [9, 222, 2509, 27, 9, 2509, 1837, 20, 125, 2509, 1348, 4, 1], [81, 549, 36, 5034, 16, 9, 1809, 3484, 24, 438, 9, 1303, 4920, 4, 1], [81, 1955, 5, 26280, 18, 9, 20524, 2213, 307, 8, 9, 1194, 4, 1]]
lens = [len(caption) for caption in list]
max_len = max(lens)
batch_size = len(list)
new_array = nd.zeros((batch_size,max_len),dtype='int')
for i in range(batch_size):
    new_array[i,:lens[i]] = list[i]
print(new_array)
lens = nd.array(lens)
elmo_intro_file = 'elmo_intro.txt'
with io.open(elmo_intro_file, 'w', encoding='utf8') as f:
    f.write(elmo_intro)

dataset = nlp.data.TextLineDataset(elmo_intro_file, 'utf8')

tokenizer = nlp.data.NLTKMosesTokenizer()
dataset = dataset.transform(tokenizer)
dataset = dataset.transform(lambda x: ['<bos>'] + x + ['<eos>'])
print(dataset[2])  # print the same tokenized sentence
#counter = nlp.data.count_tokens([word for sentence in dataset for word in sentence])
#vocab = nlp.Vocab(counter)


lstm, vocab = nlp.model.get_model('standard_lstm_lm_1500',
                                dataset_name='wikitext-2',
                                pretrained=True,
                                ctx=context[0])
print(lstm)

dataset = dataset.transform(lambda x: (vocab[x], len(x)), lazy=False)
batch_size = 3
dataset_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(),
                                              nlp.data.batchify.Stack())
data_loader = gluon.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    batchify_fn=dataset_batchify_fn)


class CaptionEncoder(gluon.HybridBlock):
    """Network for sentiment analysis."""
    def __init__(self, prefix=None, params=None):
        super(CaptionEncoder, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None # will set with lm embedding later
            self.encoder = None # will set with lm encoder later
            self.begin_state = None


    def hybrid_forward(self, F, data,hiddens, valid_length): # pylint: disable=arguments-differ
        encoded,hiddens = self.encoder(self.embedding(data),hiddens)  # Shape(T, N, C)
        return encoded,hiddens

new_model = CaptionEncoder()
new_model.embedding = lstm.embedding
new_model.encoder = lstm.encoder
new_model.begin_state = lstm.begin_state
new_model.hybridize()


print(new_model)

def get_features(data, valid_lengths):
    #length = data.shape[1]
    batch_size = data.shape[0]
    hidden_state = new_model.begin_state(func=mx.nd.zeros, batch_size=batch_size,ctx=context[0])
    #mask = mx.nd.arange(length).expand_dims(0).broadcast_axes(axis=(0,), size=(batch_size,))
    #mask = mask < valid_lengths.expand_dims(1).astype('float32')
    print(data.shape)
    data = mx.nd.transpose(data)
    o = lstm.embedding(data)
    #print(o.shape)
    output,hidden= lstm.encoder(o,hidden_state)
    print(output.shape)
    print(hidden[0].shape)
    return (output,hidden)

#batch = next(iter(data_loader))

data = new_array.as_in_context(context[0])
length = lens.as_in_context(context[0])
features,hiddens = get_features(data,length)
print([x.shape for x in hiddens])

