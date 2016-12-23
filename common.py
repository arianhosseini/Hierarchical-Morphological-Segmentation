import cPickle, os
from collections import OrderedDict

import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


def init_two_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = numpy.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)
    return rng_numpy, rng_theano

rng_numpy, rng_theano = init_two_rngs()


def init_tparams(params):
    """initialize theano shared variables"""
    tparams = OrderedDict()
    for k, v in params.iteritems():
        tparams[k] = theano.shared(params[k], k)
    return tparams

def init_norm_weight(shape, scale=0.01):
    w = scale * rng_numpy.randn(*shape)
    return w.astype(theano.config.floatX)

def init_vector_weight(dim):
    return numpy.zeros(dim,dtype=theano.config.floatX)
