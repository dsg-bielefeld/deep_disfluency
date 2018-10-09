# TODO class should be eliminated, instead factored into elman

import theano
import numpy
import os
import sys
import time

from theano import tensor as T
from collections import OrderedDict

dtype = T.config.floatX  # @UndefinedVariable


class model_no_pos(object):

    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,
                                                            (ne + 1, de)).astype(dtype))  # add one for PADDING at the end
        self.Wx = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,
                                                           (de * cs, nh)).astype(dtype))
        self.Wh = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,
                                                           (nh, nh)).astype(dtype))
        self.W = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,
                                                          (nh, nc)).astype(dtype))
        self.bh = theano.shared(numpy.zeros(nh, dtype=dtype))
        self.b = theano.shared(numpy.zeros(nc, dtype=dtype))
        self.h0 = theano.shared(numpy.zeros(nh, dtype=dtype))

        # Weights for L1 and L2
        self.L1_reg = 0.0
        self.L2_reg = 0.00001

        # bundle
        self.params = [self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0]
        self.names = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        self.idxs = T.imatrix()  # as many columns as context window size/lines as words in the sentence

        x = self.emb[self.idxs].reshape((self.idxs.shape[0], de * cs))
        self.y = T.iscalar('y')  # label

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x, outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1, 0, :]
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # TODO adding this- zero one loss for the last word
        y_pred_word = T.argmax(p_y_given_x_lastword)

        # cost and gradients and learning rate
        self.lr = T.scalar('lr')

        # Standard nll loss
        self.nll = -T.mean(T.log(p_y_given_x_lastword)[self.y])
        self.sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                                    [T.arange(x.shape[0]), self.y])

        # theano functions
        self.classify = theano.function(inputs=[self.idxs], outputs=y_pred)

        # regularisation terms
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        # if not using this set this to 0 to avoid unecessary computation
        self.L1 = 0
        #self.L1 = abs(self.Wh.sum()) +  abs(self.Wx.sum()) +  abs(self.W.sum()) +  abs(self.emb.sum())\
        #            +  abs(self.bh.sum()) + abs(self.b.sum()) + abs(self.h0.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.Wh ** 2).sum() + (self.Wx ** 2).sum() + (self.W ** 2).sum() + (self.emb ** 2).sum()\
            + (self.bh ** 2).sum() + (self.b ** 2).sum() + (self.h0 ** 2).sum()

        def errors():
            """Return a float representing the number of errors in the minibatch
            over the total number of examples of the minibatch ; zero one
            loss over the size of the minibatch

            :type y: theano.tensor.TensorType
            :param y: corresponds to a vector that gives for each example the
                      correct label
            """

            # check if y has same dimension of y_pred
            if self.y.ndim != y_pred_word.ndim:
                raise TypeError('y should have the same shape as self.y_out',
                                ('y', self.y.type, 'y_pred', y_pred_word.type))
            # check if y is of the correct datatype
            if self.y.dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                # return T.mean(T.neq(y_pred_word, self.y)) #NOTE, this shouldn't be mean if just one variable? or .. could be more..?
                return T.mean(T.neq(y_pred_word, self.y))
            else:
                raise NotImplementedError()

        # TODO what's the difference between weighted F-loss and zero-one error when we're only looking at one word?
        #self.f_loss = precision_recall_fscore_support(y_pred, self.y, average='weighted')[2]
        self.f_loss = errors()

        # the gradients to compute
        #gradients = T.grad( self.nll, self.params )
        # TODO am changing to below

        cost = self.nll \
            + self.L1_reg * self.L1 \
            + self.L2_reg * self.L2_sqr
        gradients = T.grad(cost, self.params)

        self.updates = OrderedDict((p, p - self.lr * g) for p, g in zip(self.params, gradients))

        # theano functions
        self.soft_max = theano.function(inputs=[self.idxs], outputs=p_y_given_x_sentence)  # simply outputs the soft_max distribution for each word in utterance

        #=======================================================================
        # #ORIGINAL CODE
        # self.train = theano.function( inputs  = [self.idxs, self.y, lr],
        #                               outputs = nll,
        #                               updates = self.updates )
        #=======================================================================

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb / T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})

    def fit(self, my_seq, my_indices, my_labels, lr, nw):
        tic = time.time()
        corpus, labels = self.shared_dataset(my_seq, my_labels)  # loads data set as a shared variable
        # TODO new effort to index the shared vars
        batchstart = T.iscalar('batchstart')
        batchstop = T.iscalar('batchstop')
        cost = self.nll
        # try with regularisation:
        #cost = self.nll \
        #    + self.L1_reg * self.L1 \
        #    + self.L2_reg * self.L2_sqr
        # Try with floss
        #cost = self.f_loss
        updates = self.updates
        self.train_by_index = theano.function(inputs=[batchstart, batchstop, self.lr],
                                              outputs=cost,
                                              updates=updates,
                                              givens={self.idxs: corpus[batchstart:batchstop + 1],
                                                      self.y: labels[batchstop]},
                                              on_unused_input='warn')

        # trying Theano scan now
        #=======================================================================
        # indices = T.imatrix('indices')
        # costs, newupdates = theano.scan(sequences=[indices],
        #                                non_sequences = [self.lr],
        #                                outputs = self.train_by_index
        #                                )
        #
        # cost = costs[-1]
        #
        # power = theano.function(inputs=[indices,self.lr], outputs=cost, updates=newupdates)
        #
        # return f(my_indices)
        #=======================================================================

        i = 0
        train_loss = 0.0
        for start, stop in my_indices:
            i += 1
            x = self.train_by_index(start, stop, lr)
            train_loss += x
            # print i
            self.normalize()
            if i % 6500 == 0 and i > 0:
                print '[learning] >> %2.2f%%' % ((i + 1) * 100. / nw), 'completed in %.2f (sec) <<\r' % (time.time() - tic),
                sys.stdout.flush()
        print "train_loss (may include reg)"
        print train_loss / float(i)
        return train_loss / float(i)

    def corpus_nll(self, my_seq, my_indices, my_labels):
        """Computes the average loss (per word or per utterance) for whole corpus"""
        # batch sizes is a list the same length as losses
        # which has size of each batch in it (all of sizefrom 1 to bptt limit). Computes with myindices
        #batch_sizes = [(my_indices[i][1]-my_indices[i][0])+1 for i in range(0,len(my_indices))]
        # print len(batch_sizes) #TODO don't actually need this for weighting the loss as this done anyway I believe by length of context sequence?

        corpus, labels = self.shared_dataset(my_seq, my_labels)  # loads data set as a shared variable
        batchstart = T.iscalar('batchstart')
        batchstop = T.iscalar('batchstop')
        cost = self.nll

        self.error_by_index = theano.function(inputs=[batchstart, batchstop],
                                              outputs=cost,
                                              givens={self.idxs: corpus[batchstart:batchstop + 1],
                                                      self.y: labels[batchstop]},
                                              on_unused_input='warn')
        losses = [self.error_by_index(start, stop) for start, stop in my_indices]

        return numpy.average(losses)

    def shared_dataset(self, mycorpus, mylabels, borrow=True):
        """ Load the dataset into shared variables """
        corpus = theano.shared(numpy.asarray(mycorpus, dtype='int32'),
                               borrow=True)
        # b_indices = theano.shared(numpy.asarray(myindices, dtype='int32'),
        #                          borrow=True)
        labels = theano.shared(numpy.asarray(mylabels, dtype='int32'),
                               borrow=True)
        return corpus, labels

    def load(self, folder):
        emb = numpy.load(os.path.join(folder, 'embeddings.npy'))
        Wx = numpy.load(os.path.join(folder, 'Wx.npy'))
        Wh = numpy.load(os.path.join(folder, 'Wh.npy'))
        W = numpy.load(os.path.join(folder, 'W.npy'))
        bh = numpy.load(os.path.join(folder, 'bh.npy'))
        b = numpy.load(os.path.join(folder, 'b.npy'))
        h0 = numpy.load(os.path.join(folder, 'h0.npy'))
        return emb, Wx, Wh, W, bh, b, h0

    def load_weights(self, emb=None, Wx=None, Wh=None, W=None, bh=None, b=None,
                     h0=None):
        print "loading previous weights"
        if emb is not None:
            self.emb.set_value(emb)
        if Wx is not None:
            self.Wx.set_value(Wx)
        if Wh is not None:
            self.Wh.set_value(Wh)
        if W is not None:
            self.W.set_value(W)
        if bh is not None:
            self.bh.set_value(bh)
        if b is not None:
            self.b.set_value(b)
        if h0 is not None:
            self.h0.set_value(h0)

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
