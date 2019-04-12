import warnings
warnings.filterwarnings('ignore')
import mxnet as mx
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp

class CaptionEncoder(gluon.HybridBlock):
    """Network for sentiment analysis."""

    def __init__(self, prefix=None, params=None):
        super(CaptionEncoder, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None  # will set with lm embedding later
            self.encoder = None  # will set with lm encoder later

    def hybrid_forward(self, F, data, hiddens):  # pylint: disable=arguments-differ
        encoded, hiddens = self.encoder(self.embedding(data), hiddens)  # Shape(T, N, C)
        return encoded, hiddens