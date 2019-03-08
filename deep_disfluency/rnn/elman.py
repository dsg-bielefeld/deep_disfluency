import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
import numpy as np
import os

from theano import tensor as T
from collections import OrderedDict

# nb might be theano.config.floatX
dtype = T.config.floatX  # @UndefinedVariable


class Elman(object):

    def __init__(self, ne, de, na, nh, n_out, cs, npos,
                 update_embeddings=True):
        '''
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        na :: number of acoustic or language model features at each word step
        (acoustic context size in frames * number of features)
        nh :: dimension of the hidden layer
        n_out :: number of classes
        cs :: word window context size
        npos :: number of pos tags
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                         (ne + 1, de)).
                                 astype(dtype))  # add one for PADDING
        if na == 0:
            # NB original one, now Wx becomes much bigger with acoustic data
            self.Wx = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                            ((de * cs) +
                                                             (npos * cs),
                                                             nh))
                                    .astype(dtype))
        else:
            self.Wx = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                            ((de * cs) +
                                                             (npos * cs) +
                                                             na, nh))
                                    .astype(dtype))
        self.Wh = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                        (nh, nh))
                                .astype(dtype))
        self.W = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                       (nh, n_out))
                               .astype(dtype))
        self.bh = theano.shared(np.zeros(nh, dtype=dtype))
        self.b = theano.shared(np.zeros(n_out, dtype=dtype))
        self.h0 = theano.shared(np.zeros(nh, dtype=dtype))
        # Use the eye function (diagonal 1s) for the POS, small in memory
        self.pos = T.eye(npos, npos, 0)
        self.n_acoust = na  # the number of acoustic features

        # Weights for L1 and L2
        self.L1_reg = 0.0
        self.L2_reg = 0.00001

        # without embeddings updates
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0]
        self.names = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        if update_embeddings:
            self.params = [self.emb, self.Wx, self.Wh, self.W, self.bh,
                           self.b, self.h0]
            self.names = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        # as many columns as context window size/lines as words in the sentence
        self.idxs = T.imatrix()
        self.pos_idxs = T.imatrix()

        # simply a matrix: number of features * length sentence
        self.extra_features = T.matrix()

        # TODO Old version no pos
        # x = self.emb[self.idxs].reshape((self.idxs.shape[0], de*cs))

        if na == 0:
            # POS version, not just the embeddings
            # but with the POS window concatenated
            x = T.concatenate((self.emb[self.idxs].reshape((self.idxs.shape[0],
                                                            de*cs)),
                               self.pos[self.pos_idxs].reshape(
                                                    (self.pos_idxs.shape[0],
                                                     npos*cs))), 1)
        else:
            # TODO new version with extra features
            x = T.concatenate((self.emb[self.idxs].reshape((self.idxs.shape[0],
                                                            de*cs)),
                               self.pos[self.pos_idxs].reshape(
                                                    (self.pos_idxs.shape[0],
                                                     npos*cs)),
                               self.extra_features), 1)
        self.y = T.iscalar('y')  # label
        # TODO for sentences
        # self.y = T.ivector('y') #labels for whole sentence

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) +
                                 self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x, outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1, 0, :]
        p_y_given_x_sentence = s[:, 0, :]
        p_y_given_x_sentence_hidden = (h, s[:, 0, :])
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # TODO adding this- zero one loss for the last word
        # y_pred_word = T.argmax(p_y_given_x_lastword)

        # learning rate not hard coded as could decay
        self.lr = T.scalar('lr')

        # Cost: standard nll loss
        self.nll = -T.mean(T.log(p_y_given_x_lastword)[self.y])
        self.sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                                    [T.arange(x.shape[0]), self.y])

        if na == 0:
            self.classify = theano.function(inputs=[self.idxs, self.pos_idxs],
                                            outputs=y_pred)
        else:
            self.classify = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                    self.extra_features],
                                            outputs=y_pred)

        # regularisation terms
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        # if not using this set this to 0 to avoid unecessary computation
        self.L1 = 0
        # self.L1 = abs(self.Wh.sum()) +  abs(self.Wx.sum()) +  \
        # abs(self.W.sum()) +  abs(self.emb.sum())\
        # +  abs(self.bh.sum()) + abs(self.b.sum()) + abs(self.h0.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.Wh ** 2).sum() + (self.Wx ** 2).sum() +\
                      (self.W ** 2).sum() + (self.emb ** 2).sum() +\
                      (self.bh ** 2).sum() + (self.b ** 2).sum() +\
                      (self.h0 ** 2).sum()

        self.cost = self.nll \
            + self.L1_reg * self.L1 \
            + self.L2_reg * self.L2_sqr
        gradients = T.grad(self.cost, self.params)

        self.updates = OrderedDict((p, p-self.lr*g)
                                   for p, g in zip(self.params, gradients))

        # costs for multiple labels (one for each in the input)
        self.sentence_cost = self.sentence_nll \
            + self.L1_reg * self.L1 \
            + self.L2_reg * self.L2_sqr
        sentence_gradients = T.grad(self.sentence_cost, self.params)

        self.sentence_updates = OrderedDict((p, p - self.lr*g)
                                            for p, g in
                                            zip(self.params,
                                                sentence_gradients))

        if na == 0:
            self.soft_max = theano.function(inputs=[self.idxs, self.pos_idxs],
                                            outputs=p_y_given_x_sentence)
            self.soft_max_return_hidden_layer = theano.function(
                                    inputs=[self.idxs, self.pos_idxs],
                                    outputs=p_y_given_x_sentence_hidden)
        else:
            self.soft_max = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                    self.extra_features],
                                            outputs=p_y_given_x_sentence)
            self.soft_max_return_hidden_layer = theano.function(
                                        inputs=[self.idxs, self.pos_idxs,
                                                self.extra_features],
                                        outputs=p_y_given_x_sentence_hidden)

        if na == 0:
            self.train = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                 self.y,
                                                 self.lr],
                                         outputs=self.nll,
                                         updates=self.updates)
        else:
            self.train = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                 self.extra_features,
                                                 self.y,
                                                 self.lr],
                                         outputs=self.nll,
                                         updates=self.updates)

        self.normalize = theano.function(
                        inputs=[],
                        updates={self.emb:
                                 self.emb /
                                 T.sqrt((self.emb**2).sum(axis=1))
                                 .dimshuffle(0, 'x')}
                                         )

    def classify_by_index(self, word_idx, indices, pos_idx=None,
                          extra_features=None):
        """Classification method which assumes the dialogue matrix is
        in the right format.

        :param word_idx: window size * dialogue length matrix
        :param labels: vector dialogue length long
        :param indices: 2 * dialogue length matrix for start, stop indices
        :param pos_idx: pos window size * dialogue length matrix
        :param extra_features: number of features * dialogue length matrix
        """
        output = []
        for start, stop in indices:

            if extra_features:

                output.extend(self.classify(word_idx[start:stop+1, :],
                                            pos_idx[start:stop+1, :],
                                            np.asarray(
                                            extra_features[start:stop+1, :],
                                            dtype='float32')
                                            )
                              )
            else:
                output.extend(self.classify(word_idx[start:stop+1, :],
                                            pos_idx[start:stop+1, :]
                                            )
                              )
        return output

    def fit(self, word_idx, labels, lr, indices, pos_idx=None,
            extra_features=None):
        """Fit method which assumes the dialogue matrix is in the right
        format.

        :param word_idx: window size * dialogue length matrix
        :param labels: vector dialogue length long
        :param indices: 2 * dialogue length matrix for start, stop indices
        :param pos_idx: pos window size * dialogue length matrix
        :param extra_features: number of features * dialogue length matrix
        """
        loss = 0
        test = 0
        testing = False
        for start, stop in indices:
            # print start, stop
            if testing:
                test += 1
                if test > 50:
                    break
            if extra_features:

                x = self.train(word_idx[start:stop+1, :],
                               pos_idx[start:stop+1, :],
                               np.asarray(extra_features[start:stop+1, :],
                                          dtype='float32'),
                               labels[stop],
                               lr)
            else:
                x = self.train(word_idx[start:stop+1, :],
                               pos_idx[start:stop+1, :],
                               labels[stop],
                               lr)
            loss += x
            self.normalize()
        return loss

    def shared_dataset(self, mycorpus, borrow=True, data_type='int32'):
        """ Load the dataset into shared variables """
        return theano.shared(np.asarray(mycorpus, dtype=data_type),
                             borrow=True)

    def load_weights_from_folder(self, folder):
        for name, param in zip(self.names, self.params):
            param.set_value(np.load(os.path.join(folder, name + ".npy")))

    def load(self, folder):
        emb = np.load(os.path.join(folder, 'embeddings.npy'))
        Wx = np.load(os.path.join(folder, 'Wx.npy'))
        Wh = np.load(os.path.join(folder, 'Wh.npy'))
        W = np.load(os.path.join(folder, 'W.npy'))
        bh = np.load(os.path.join(folder, 'bh.npy'))
        b = np.load(os.path.join(folder, 'b.npy'))
        h0 = np.load(os.path.join(folder, 'h0.npy'))
        return emb, Wx, Wh, W, bh, b, h0

    def load_weights(self, emb=None, Wx=None, Wh=None, W=None, bh=None, b=None,
                     h0=None):
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
            np.save(os.path.join(folder, name + '.npy'), param.get_value())
