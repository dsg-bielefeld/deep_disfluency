# Ngrams models
from __future__ import division
import math
from collections import defaultdict
import cPickle
from operator import itemgetter
import numpy

from util import safe_open, flush_and_close

infinity = float('inf')
minus_infinity = - infinity
tag_separator = '_@'
log_base = 2  # for constency with nltk

# this function is needed just to use the defaultdict container


def defaultdict_int_factory():
    """just a wrapper around a call to defaultdict(int)"""
    return defaultdict(int)


def properPrefix(pref, word):
    """removes the $unk$ tag on proper nouns and numbers"""
    word = word.replace("$unk$", "")
    pref = pref.replace("$unk$", "")
    if word == "" or pref == "":
        return False
    if word.startswith(pref) and not word == pref:
        return True
    return False


def log(x):
    """A wrapper around math.log10, so that it returns 0
    in the case the values is 0."""
    if x == 0.0:
        return 0
    else:
        return math.log(x, log_base)  # for consistency with nltk


class LanguageModel(object):
    """A generic language model that actually doesn't do anything"""

    def save(self, index_file):
        """Saves a model to a file"""
        raise NameError('Method save not implemented')

    @classmethod
    def load(klass, attrs, ngrams=None):
        """Loads a model from the JSON attributes parsed from an index file"""
        raise NameError('Method corpus not implemented')

    def repopulate_from_attrs(self, attrs, ngrams=None):
        """Replaces the content of the current model using attrs"""
        raise NameError('Method repopulate_from_attrs not implemented')

    def tokenize_sentence(self, sentence, order):
        """Returns a list of tokens with the correct numbers of initial
        and end tags (this is meant ot be used with a non-backoff model!!!)"""
        tokens = sentence.split()
        tokens = ['<s>'] * (order - 1) + tokens + ['</s>']
        return tokens

    def glue_tokens(self, tokens):
        """The standard way in which we glue together tokens to create keys
        in our hashmaps"""
        return ' '.join(tokens)

    def unglue_tokens(self, tokens):
        """Converse of glue_tokens"""
        return tokens.split()

    def train(self, train_corpus, order):
        """This method creates the model by reading data from a corpus
        (a file like object open for reading) and trains a model
        of the given order"""
        raise NameError('Method train not implemented')

    def ngram_prob(self, ngram, order):
        """Returns the probability of a single ngram (probability, not
        its log). The ngram must be a list of tokens"""
        raise NameError('Method ngram_prob not implemented')

    def tokens_logprob(self, tokens, order, special=None):
        """Returns the logprobs assigned by the model to each token for
        the specified order.
        The method skips the first order-1 tokens as initial context."""
        delta = self.order - 1
        return [log(self.ngram_prob(tokens[i - delta: i + 1], order, special))
                for i in range(delta, len(tokens))]

    def tokens_logprob_backoff(self, tokens, order, special=None):
        """Returns the logprobs assigned by the model to each token for the
        specified order.
        The method skips the first order-1 tokens as initial context.
        Uses the backoff probability too."""
        delta = self.order - 1
        if order == self.order:
            return [log(self.ngram_prob(tokens[i - delta: i + 1],
                                        order, special))
                    for i in range(delta, len(tokens))]
        delta = order - 1
        return [log(self.ngram_prob(tokens[i - delta: i + 1], order, special))
                for i in range(delta, len(tokens))]

    def logprob(self, sentence):
        """Returns the logprob of a sentence"""
        tokens = self.tokenize_sentence(sentence, self.order)
        return sum(self.tokens_logprob(tokens, self.order))

    def surprisal(self, tokens, order):
        p = self.ngram_prob(tokens, order)
        return (- log(p))

    def weighted_surprisal(self, tokens, order):
        p = self.ngram_prob(tokens, order)
        return p * (- log(p))

    def entropy(self, text, order):
        """This method calculates the entropy score for a text.
        The text is tokenized by this method (and the necessary <s> and
        </s> tags added)"""
        tokens = self.tokenize_sentence(text, order)
        s = 0.
        delta = self.order - 1
        for i in xrange(delta, len(tokens)):
            ng = tokens[i - delta: i + 1]
            p = self.ngram_prob(ng, order)  # got a partial word factor
            s += (p * -log(p))
        return s

    def perplexity(self, text):
        """This method calculates the perplexity score for a text.
        The text is tokenized by this method
        (and the necessary <s> and </s> tags added)"""
        return log_base ** self.entropy(text)

    def cross_entropy(self, text, order):
        # inverse of sentence normalised log prob
        tokens = text.split()
        s = 0
        for i in range(order - 1, len(tokens)):
            print tokens[i - order + 1:i + 1]
            p = self.surprisal(tokens[i - order + 1:i + 1], order)
            print p
            s += p
        return s * (1.0 / float(len(tokens)))

    def logprob_weighted_by_sequence_length(self, tokens):
        """Returns the sum of the n-grams logprobs divided by the number
        of tokens"""
        return sum(self.tokens_logprob(tokens, self.order)) / len(tokens)

    def logprob_weighted_by_inverse_unigram_logprob(self, tokens, special=None):
        """WML in Clark et al. 2013. The logprob sum weighted by inverse of
        the sum of the unigram logprobs.
        Returns the sum of the n-grams logprobs divided by the inverse of
        the sum of the unigram probabilities"""
        test = sum(self.tokens_logprob(tokens, 1, special=special))
        if test == 0:
            print tokens
        return sum(self.tokens_logprob(tokens, self.order, special)) / \
            (-1. * test)

    def normalized_min_logprob(self, tokens):
        """Returns the lowest logprob assigned to the n-grams of the sentence
        divided by its unigram logprob"""
        logprobs = self.tokens_logprob(tokens, self.order)
        unigram_logprobs = self.tokens_logprob(tokens, 1)
        return min(map(lambda x, y: - (x / y), logprobs, unigram_logprobs))

    def normalized_max_logprob(self, tokens):
        """Returns the highest logprob assigned to the n-grams of the sentence
        divided by its unigram logprob"""
        logprobs = self.tokens_logprob(tokens, self.order)
        unigram_logprobs = self.tokens_logprob(tokens, 1)
        return max(map(lambda x, y: - (x / y), logprobs, unigram_logprobs))

    def mean_of_n_smallest_ngrams(self, tokens, n=2):
        """Returns the mean value of the n lowest scoring n-grams (normalized
        by their unigram logprob)"""
        logprobs = self.tokens_logprob(tokens, self.order)
        unigram_logprobs = self.tokens_logprob(tokens, 1)
        return sum(sorted(map(lambda x, y: - (x / y),
                              logprobs, unigram_logprobs))[:n]) / n

    def prob_fluent(self, tokens, order):
        raise NameError("Method p_fluent not implemented")

    def prob_distribution_over_completions(self, context):
        """will give us the prob_distribution_over_completions of possible
        continuations from these context tokens"""
        raise NameError('Method not implemented')

    def entropy_continuation(self, tokens, order):
        """This method calculates the entropy of the
        prob_distribution_over_completions of following words given
        the context tokens,
        This may only be useful for n-1 context lengths as we don't
        propogate (though see Clark et al 13) a two-word context,
        though this is generalisable
        """
        raise NameError('Method entropy_continuation not implemented')

    def cache_result(self, ngram, mytuple):
        """simple caching procedure that should help speed things up"""
        self.cache[str(ngram)] = mytuple
        if len(self.cache.keys()) > self.cache_limit:
            for key in self.cache.iterkeys():  # removes any other element
                if not key == str(ngram):
                    del self.cache[key]
                    break

    def init_cache(self):
        """"Initializes a cache to speed access up"""
        self.cache = defaultdict(float)
        self.cache_limit = 300  # keeps most common n sparse matrices


class NgramGraph(object):
    """DAG structure that stores the current probability of the whole sequence
    and the prob of its unigrams
    for fast computation of the liklihood of subsequences without
    recomputation.
    i.e. it allows word-by-word incremental decoding of current sequence)"""

    def __init__(self, lm, words=None):
        self.delta = lm.order - 1
        self.prob_graph = self.delta * [0]  # won't make a difference
        self.unigram_graph = self.delta * [0]
        self.total_logprob = self.delta * [0]
        self.total_unigram_logprob = self.delta * [0]
        self.word_graph = self.delta * ["<s>"]  # initialise
        if words != None:  # can initialise with words
            for word in words[self.delta:]:
                self.append(word, lm)  # will calculate all this automatically?

    def subgraph(self, indices, lm):
        """return a copy of the graph acc the indices, first one inclusive,
        second one exclusive"""
        new = NgramGraph(lm)
        new.word_graph = list(self.word_graph[indices[0]:indices[1]])
        new.prob_graph = list(self.prob_graph[indices[0]:indices[1]])
        new.unigram_graph = list(self.unigram_graph[indices[0]:indices[1]])
        new.total_logprob = list(self.total_logprob[indices[0]:indices[1]])
        new.total_unigram_logprob = list(
            self.total_unigram_logprob[indices[0]:indices[1]])
        return new

    def append(self, word, lm):
        """Add a word to the list (right frontier, calculate its prob with
        its context"""
        # print "appending"
        # print word
        self.word_graph.append(word)
        # print self.word_graph[-lm.order:]
        newngram = lm.ngram_prob(self.word_graph[-lm.order:], lm.order)
        self.prob_graph.append(newngram)
        self.total_logprob.append(self.total_logprob[-1] + log(newngram))
        newunigram = lm.ngram_prob([self.word_graph[-1]], 1)
        self.unigram_graph.append(newunigram)
        self.total_unigram_logprob.append(self.total_unigram_logprob[-1] +
                                          log(newunigram))

    def extend(self, graph, lm):  # sticking two together
        self.word_graph += list(graph.word_graph)  # initialise
        for unigram in graph.unigram_graph:
            # print unigram
            self.total_unigram_logprob.append(
                self.total_unigram_logprob[-1] + log(unigram))
        self.unigram_graph += list(graph.unigram_graph)
        for i in range(len(self.word_graph) - len(graph.word_graph),
                       len(self.word_graph)):
            newngram = lm.ngram_prob(self.word_graph[i - self.delta:i + 1],
                                     lm.order)
            self.prob_graph.append(newngram)
            # print "prob graph append"
            # print self.word_graph[i-self.delta:i+1]
            self.total_logprob.append(self.total_logprob[-1] + log(newngram))

    def logprob_weighted_by_sequence_length(self, indices=None):
        if indices == None:
            return float(self.total_logprob[-1]) / float(len(self.word_graph))
        elif len(indices) == 1:  # just specifying end
            return float(self.total_logprob[0:indices[0]]) / float(indices[0])
        else:  # specifiying start and end
            # pass #TODO
            return float(self.total_logprob[indices[1]] -
                         self.total_logprob[indices[0]]) / \
                float(indices[1] - indices[0])

    def logprob_weighted_by_inverse_unigram_logprob(self, indices=None):
        """indices is an int list of length 1 or 2, if 1, just end point,
        if two, start and end point"""
        if indices == None:
            return float(self.total_logprob[-1]) / \
                float(-1.0 * self.total_unigram_logprob[-1])
        elif (len(indices) == 1):
            return float(self.total_logprob[indices[0]]) / \
                float(-1.0 * self.total_unigram_logprob[indices[0]])
        else:
            return float(self.total_logprob[indices[1]] -
                         self.total_logprob[indices[0]]) / \
                float(-1.0 *
                      float(self.total_unigram_logprob[indices[1]] -
                            self.total_unigram_logprob[indices[0]]))


class KneserNeySmoothingModel(LanguageModel):
    """The standard implementation of Kneser Ney interpolation.
    This has specialized decoding, training, loading and saving methods
    for standard KN."""

    def __init__(self, order=3, discount=None, partial_words=False,
                 train_corpus=None, heldout_corpus=None, second_corpus=None,
                 verbose=True, saved_file=None):
        """
        Keyword arguments:

        train_corpus -- a corpus file or string of words,
        which if passed to the constructor the model is
        trained right away (default None)
        heldout_corpus -- a smaller corpus file or string which treats
        words not in vocab so far as unknown (default None)
        second_corpus -- if present, a corpus file or string which is
        an additional training source
        order -- the order of the model (3 = trigram, 2 = bigram, etc.)
        (default 3)
        partial_words -- boolean as to whether this deals with partial words
        with special probs or not
        verbose -- whether you want online training timing output or not
        """

        self.order = order
        self.partial_words = partial_words
        self.discount = discount
        self.verbose = verbose
        self.train_length = 0  # gets initialized
        # to check the training corpus length

        # params of the model which can be loaded or learned
        self.unigrams = []  # Just an ordered list of unigrams
        self.unigram_counts = defaultdict(int)
        self.bigrams = []  # Essentially just a subset of ngram_denom
        self.bigram_counts = defaultdict(int)

        self.bigram_types = 0
        self.trigram_types = 0
        self.vocab_size = 0
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.unigram_denominator = 0  # for unigrams we don't have contexts
        self.ngram_non_zero_map = defaultdict(int)
        self.unigram_contexts = defaultdict(list)
        self.bigram_contexts = defaultdict(list)

        self.bigram_history_entropies = defaultdict(float)  # Omitting these
        self.trigram_history_entropies = defaultdict(float)

        if train_corpus != None:
            self.train(train_corpus)
        if second_corpus != None:
            print "using second corpus"
            self.train(second_corpus)
        if heldout_corpus != None:
            print "using heldout corpus"
            self.train(heldout_corpus, heldout=True)

        if train_corpus != None:  # this is required to
            # order the unigrams and bigrams for entropies,
            # can probably discard after entropy cacheing,
            print str(self.train_length), "total words of training data"
            for ngram, val in sorted(self.unigram_counts.items(),
                                     key=itemgetter(1), reverse=True):
                                                                            self.unigrams.append(ngram)
            self.vocab_size = len(self.unigrams)
            for ngram, val in sorted(self.bigram_counts.items(),
                                     key=itemgetter(1), reverse=True):
                                                                            self.bigrams.append(ngram)
            self.init_entropy_cache(0.40)  # initialise max ent

        # if this gets loaded instead of trained,
        # populates everything from pickled db
        if saved_file != None:
            self.load(saved_file)

        # self.init_cache() #TODO we could look at cacheing
        print "1-grams =", str(self.vocab_size)
        print "2-grams =", str(self.bigram_types)
        print "3-grams =", str(self.trigram_types)
        # print self.unigrams

    def glue_tokens(self, tokens, order):
        """NB this is specific to the way we store these for \
        Kneser Ney smoothing"""
        return '{0}@{1}'.format(order, ' '.join(tokens))

    def unglue_tokens(self, tokenstring, order):
        if order == 1:
            return [tokenstring.split("@")[1].replace(" ", "")]
        return tokenstring.split("@")[1].split(" ")

    def ngrams_interpolated_kneser_ney(self, tokens, order):
        """This function counts the n-grams in tokens and also record the
        lower order non zero counts necessary for interpolated Kneser-Ney \
        smoothing,
        taken from Goodman 2001 and generalized to arbitrary orders"""
        l = len(tokens)
        for i in xrange(order - 1, l):  # tokens should have a prefix of order - 1
            # print i
            for d in xrange(order, 0, -1):  # go through all the different 'n's
                if d == 1:
                    self.unigram_denominator += 1
                    # print "unigram_denom" + str(self.unigram_denominator)
                    self.ngram_numerator_map[self.glue_tokens(tokens[i], d)]\
                        += 1
                    # print self.ngram_numerator_map
                    # raw_input()
                else:
                    den_key = self.glue_tokens(tokens[i - (d - 1): i], d)
                    num_key = self.glue_tokens(tokens[i - (d - 1): i + 1], d)
                    # print den_key
                    # print num_key
                    if d == self.order:
                        # just takes the last one to get the unigram count
                        self.unigram_counts[tokens[i]] += 1
                        self.bigram_counts[' '.join(tokens[i - 1:i + 1])] += 1
                        # a raw count of the bigrams,
                        # rather than the Kneser-Ney purely type based approach

                    self.ngram_denominator_map[den_key] += 1
                    # we store this value to check if it's 0
                    tmp = self.ngram_numerator_map[num_key]
                    self.ngram_numerator_map[num_key] += 1  # we increment it
                    if tmp == 0:  # if this is the first time we see this ngram
                        # number of types it's been used as a context for
                        self.ngram_non_zero_map[den_key] += 1
                        if d == 2:
                            # bit wasteful as essentially building
                            # the model again
                            self.unigram_contexts[tokens[i - 1]].append(
                                tokens[i])
                            self.bigram_types += 1
                        elif d == 3:
                            # this can be cleared after training
                            # as not needed after
                            # prob_distribution_over_completions creation
                            self.bigram_contexts[' '.join(
                                tokens[i - (d - 1): i])].append(tokens[i])

                            self.trigram_types += 1
                        # we increment the non zero count and
                        # implicitly switch to the lower order
                        # neccessary for interpolation coefficient
                    else:
                        break
                        # if the ngram has already been seen
                        # we don't go down to lower order models

    def train(self, train_corpus, heldout=False):
        """This method creates the model by reading data from a corpus
        (a file like object open for reading) and trains a model
        of the given order"""
        # if special mode, i.e. string, split on new line character \n
        if isinstance(train_corpus, str):
            print "training corpus is a string"
            train_corpus = train_corpus.split("\n")  # split string by new line
            is_file = False
        else:
            print "training corpus is a file"
            train_corpus.seek(0)  # we reset the corpus reading position
            is_file = True

        if heldout == True:
            totalunk = 0
            print "Training language model on heldout data..."
        else:
            print "Training language model on standard data..."

        for line in train_corpus:
            if is_file:  # assuming a REF file here
                text = line.split(",")[-1]
                # if we have a normal tag
                tokens = self.tokenize_sentence(text, self.order)
                for line in train_corpus:  # skip a line
                    break
            else:  # just string to parse
                # if we have a normal tag
                # print line
                tokens = self.tokenize_sentence(line, self.order)
            if heldout == True:
                oldtokens = list(tokens)
                tokens = []
                # put unknown token in for unknown words,
                # only form of held out est used
                for token in oldtokens:
                    if (not (token == "<s>" or token == "</s>"))\
                        and (not self.ngram_numerator_map.get(
                            self.glue_tokens(token, 1))):
                        tokens.append("<unk>")
                        totalunk += 1
                    else:
                        tokens.append(token)

            self.train_length += len(tokens) - self.order
            self.ngrams_interpolated_kneser_ney(tokens, self.order)

        self.vocab_size = len(self.unigram_counts.keys())
        # always updates after any training
        if heldout == True:
            print "unknown words in heldout data", totalunk
            if totalunk == 0:  # have to add the dummy <unk> if no unknown ones
                text = "<unk>"
                tokens = self.tokenize_sentence(text, self.order)
                self.ngrams_interpolated_kneser_ney(tokens, self.order)
        print "TOTAL WORDS TRAINED ON =", self.train_length

    def raw_ngram_prob(self, ngram, discount, order, partialWordFactor=False):
        """The internal implementation of ngram_prob.
           We could do without it but if we want, for some reason,
           to use a different discount than the general one
           we can do it with this function.
        """
        tokens = []
        # filter out unseen
        partialWord = 1
        #cache = self.cache.get(str(ngram))
        # if cache != None:
        #    #cache = self.cache[str(ngram)] #just accesses it so it's on top?
        #    return cache
        if order >= 2 and partialWordFactor == True:  # partial words
            if (ngram[-2][-1] == "-" or
                (not self.ngram_numerator_map.get(self.glue_tokens(ngram[-2], 1))
                 and properPrefix(ngram[-2], ngram[-1]) == True)):
                # print "partial"
                # print order
                # print ngram[-2]
                # raw_input()
                partialWord = 0.00001
        for token in ngram:
            # put unknown token in for unknown words, only form of held
            # out est used
            if (not self.ngram_numerator_map.get(self.glue_tokens(token, 1))) \
                    and not token == "<s>":  # i.e. never seen at all
                tokens.append("<unk>")
            else:
                if token[-1] == "-":
                    pass  # TODO add the argmax?
                tokens.append(token)
        ngram = tokens  # we've added our unk tokens

        # calculate the unigram prob of the last token
        # as it appears as a numerator
        # if we've never seen it at all, it defacto will
        # have no probability as a numerator
        uni_num = self.ngram_numerator_map.get(self.glue_tokens(ngram[-1], 1))
        if uni_num == None:
            uni_num = 0
        probability = previous_prob = float(uni_num) / \
            float(self.unigram_denominator)
        if probability == 0.0:
            print "0 prob!"
            print self.glue_tokens(ngram[-1], 1)
            print ngram
            print self.ngram_numerator_map.get(self.glue_tokens(ngram[-1], 1))
            print self.unigram_denominator
            raw_input()

        # now we compute the higher order probs and interpolate
        for d in xrange(2, order + 1):
            ngram_den = self.ngram_denominator_map.get(
                self.glue_tokens(ngram[-(d):-1], d))
            if ngram_den == None:
                ngram_den = 0
            # for bigrams this is the number of different continuation types
            # (number of trigram types with these two words)
            if ngram_den != 0:
                # if this context (bigram, for trigrams) has never been seen,
                # then we can only get unigram est, starts from two, goes up
                ngram_num = self.ngram_numerator_map.get(
                    self.glue_tokens(ngram[-(d):], d))
                # this is adding one, use get?
                if ngram_num == None:
                    ngram_num = 0
                if ngram_num != 0:
                    current_prob = (ngram_num - discount) / float(ngram_den)
                else:
                    current_prob = 0.0
                nonzero = self.ngram_non_zero_map.get(
                    self.glue_tokens(ngram[-(d):-1], d))
                if nonzero == None:
                    nonzero = 0
                current_prob += nonzero * discount / ngram_den * previous_prob
                previous_prob = current_prob
                probability = current_prob
            else:
                # current unseen contexts just give you the unigram
                # back..not ideal.. we can learn <unk> from
                # held out data though..
                probability = previous_prob
                break
        return probability * partialWord

    def ngram_prob(self, ngram, order, special=None):
        """Calculates the logprob of a single ngram.
        ngram must be a list of tokens (the order parameter is there
        just to avoid having to compute it).
        Taken from Goodman 2001 and generalized to arbitrary orders"""
        if not special == None:
            return self.raw_ngram_prob_special(ngram,
                                               self.discount, order, special)
        return self.raw_ngram_prob(ngram, self.discount, order,
                                   partialWordFactor=self.partial_words)

    def entropy_continuation(self, contexttokens, order):
        """ computes the entropy over possible completions
        Can be slow with full vocab."""
        totalMass = 0
        s = 0
        tokens = []
        for token in contexttokens:
            # put unknown token in for unknown words,
            # only form of held out est used
            if (not self.ngram_numerator_map.get(self.glue_tokens(token, 1))) \
                    and not token == "<s>":  # i.e. never seen at all
                tokens.append("<unk>")
            else:
                tokens.append(token)
        contexttokens = tokens

        for key in self.unigrams:  # LOOK AT ALL NGRAMS /or look at unigrams
            # print key
            test = str(key)
            testngram = list(contexttokens) + [test]  # NB Hashmaps are mutable
            p = self.ngram_prob(testngram, order)
            totalMass += p
            s += (p * -log(p))
        assert abs(1.0 - totalMass) <= 0.00000000001
        return s

    def entropy_continuation_very_fast(self, contexttokens, order,
                                       returneps=False, returnDist=False):
        """Computes the entropy over possible completions (over whole vocab).
        Uses the trie structure of the dicts for speed"""
        tokens = []
        for token in contexttokens:
            # put unknown token in for unknown words,
            # only form of held out est used
            if (not self.ngram_numerator_map.get(self.glue_tokens(token, 1))) \
                    and not token == "<s>":  # i.e. never seen at all
                # print "unseen"
                tokens.append("<unk>")
            else:
                tokens.append(token)
        contexttokens = tokens
        test = None
        if order == 3:
            test = self.trigram_history_entropies.get(' '.join(contexttokens))
        elif order == 2:
            test = self.bigram_history_entropies.get(contexttokens[0])
        if not test == None:
            return test
        if returnDist == True:
            pass
            # cscMatrix
        #tic = time.clock()
        totalMass = 0
        s = 0
        if order == 1:
            # print "just giving max ent"
            return self.max_ent_continuation  # i.e. smoothed entropy
        number = self.ngram_non_zero_map.get(self.glue_tokens(contexttokens,
                                                              order))
        # number of types of trigram, should be the same as below
        # print number
        if not number:
            # print "backing off"
            return self.entropy_continuation_very_fast(
                contexttokens[1:], order - 1,
                returneps=returneps,
                returnDist=returnDist)
            # otherwise go down
        check = 0

        if order == 3:
            positives = self.bigram_contexts.get(' '.join(
                contexttokens))
        if order == 2:
            positives = self.unigram_contexts.get(
                contexttokens[-1])
        for key in positives:  # LOOK AT ALL NGRAMS /or look at unigrams
            # print key
            check += 1
            test = str(key)
            testngram = list(contexttokens) + [test]  # NB Hashmaps are mutable
            p = self.ngram_prob(testngram, order)
            totalMass += p
            s += (p * -log(p))
        assert (check == number), str(check) + " " + str(number)
        # print "total mass before " + str(totalMass)
        remainder = float(1.0) - float(totalMass)
        eps = remainder / (self.vocab_size - number)
        # you cannot easily do this before hand..
        totalMass += ((self.vocab_size - number) * eps)
        s += (self.vocab_size - number) * (eps * -log(eps))
        # print time.clock()-tic
        # print "very fast continuation entropy of " + \
        #str(contexttokens) + "=" + str(s)
        # print "total mass= " + str(totalMass) +"\n"
        return s

    def init_entropy_cache(self, percent):
        """helps with storage of top M n-1 gram contexts' entropies,
        and the max ent"""
        totalmass = 0
        s = 0
        for key in self.unigrams:
            p = self.ngram_prob([key], 1)
            totalmass += p
            s += (p * -log(p))
        # print "max ent = " + str(s)
        self.max_ent_continuation = s
        assert abs(1.0 - totalmass) <= 0.00000000001, "not summing to one" + \
            str(totalmass)

        # now estimate the top x% of entropies of trigram and bigram contexts
        # do trigrams
        if self.order == 3:
            self.trigram_history_entropies["<s> <s>"] = \
                self.entropy_continuation_very_fast(["<s>", "<s>"], 3)
            target = int(float(len(self.bigrams)) * float(percent))
            for i in range(0, target):
                # not useful for this as not a context
                if self.bigrams[i].split()[-1] == "</s>":
                    continue
                self.trigram_history_entropies[self.bigrams[i]] = \
                    self.entropy_continuation_very_fast(self.bigrams[i].split(), 3)
            # print len(self.trigram_history_entropies)
        # now do bigrams
        self.bigram_history_entropies["<s>"] = \
            self.entropy_continuation_very_fast(["<s>"], 2)
        target = int(float(len(self.unigrams)) * float((percent)))
        # in reality, percent should be greater for bigrams
        for i in range(0, target):
            if self.unigrams[i] == "</s>":
                continue  # not useful for this
            self.bigram_history_entropies[self.unigrams[i]] = \
                self.entropy_continuation_very_fast([self.unigrams[i]], 2)

    def sum_information_gain_very_fast(self, tokens, order=None):
        """computes entropy of the context tokens and actual probability \
        of each ngram to give the total uncertainty reduction"""
        if order == None:
            order = self.order
        delta = order - 1
        return sum([log(-self.logprob_weighted_by_sequence_length(
            tokens[i - delta:i + 1]) /
            self.entropy_continuation_very_fast(
            tokens[i - delta:i], order))
            for i in range(delta, len(tokens))])

    def KL_divergence_continuation(self, contexttokens1, contexttokens2):
        """Can make this general, just tricky because of unique
        glue_tokens method.
            NB it's assymetric.
            Assuming normal distributions for lower order(s)- speeds
            things up a lot
            Only allow this full version for POS model as it's quick,
            need something quicker otherwise
        """
        totalMass1 = 0
        totalMass2 = 0
        KL = 0
        if (contexttokens1 == contexttokens2):
            return KL  # i.e. no divergence

        for key in self.unigrams:  # LOOK AT ALL NGRAMS /unigrams
            target = str(key)
            p1 = self.ngram_prob(contexttokens1 + [target], self.order)
            p2 = self.ngram_prob(contexttokens2 + [target], self.order)
            totalMass1 += p1
            totalMass2 += p2
            if p1 == 0:
                newKL = 0
                # raw_input("KL prob = 0!! " + str(contexttokens1) +\
                # str(target))
            elif p2 == 0:
                print "INFINITE KL DIVERGENCE!"  # todo should we
                # still check they're proper prob dists?
                return infinity
            else:
                newKL = log(p1 / p2) * p1
            KL += newKL
        assert abs(1.0 - totalMass1) <= 0.00000000001 and \
            abs(1.0 - totalMass2) <= 0.00000000001,\
            "NOT SUMMING TO 1. total mass 1 = " + str(totalMass1) +\
            " total mass 2 = " \
            + str(totalMass2) + str(contexttokens1) + str(contexttokens2)
        print "KL div" + str(contexttokens1) + str(contexttokens2) + \
            " = " + str(KL) + "\n"
        return KL

    def KL_divergence_continuation_fast(self, contexttokens1, contexttokens2):
        """Faster version of KL_divergence_continuation which
        approximates unseen counts by smoothing.
        """
        #tic = time.clock()
        totalMass1 = 0
        totalMass2 = 0
        KL = 0
        if (contexttokens1 == contexttokens2):
            # print "KLDIV 0 " + str(contexttokens1)
            return KL  # i.e. no divergence
        #if contexttokens1[-1] == contexttokens2[-1]: bigram = True
        # they have the same bigram context, will save time
        check = 0
        # vocab = [] #if not seen before, try it
        zeroCount1 = []  # the probs of context 2 + target that have zero counts
        zeroCount2 = []  # the probs of context 1 + target  that have zero counts
        bothZero = 0
        for key in self.unigrams:  # LOOK AT ALL NGRAMS /unigrams

            test = str(key)
            # assuming normal prob_distribution_over_completions
            # over unigrams only (i.e. leaving out unseen bigrams)
            zero1 = False
            #zero2 = False
            if not self.ngram_numerator_map.get(self.glue_tokens(
                    [contexttokens1[-1], test], 2)):
                zero1 = True
            if not self.ngram_numerator_map.get(self.glue_tokens(
                                                [contexttokens2[-1], test], 2)):
                #zero2 = True
                if zero1 == True:
                    bothZero += 1
                else:
                    p = self.ngram_prob(contexttokens1 + [test], self.order)
                    zeroCount2.append(p)
                    totalMass1 += p
                continue
            elif zero1 == True:
                p = self.ngram_prob(contexttokens2 + [test], self.order)
                zeroCount1.append(p)
                totalMass2 += p
                continue

            # got this far we have non-zero counts for both
            check += 1
            p1 = self.ngram_prob(contexttokens1 + [test], self.order)
            p2 = self.ngram_prob(contexttokens2 + [test], self.order)
            totalMass1 += p1
            totalMass2 += p2
            if p1 == 0:
                newKL = 0
                # raw_input("KL prob = 0!! " + str(contexttokens1) +\
                # str(target))
            elif p2 == 0:
                print "INFINITE KL DIVERGENCE!"  # TODO prob dist?
                return infinity
            else:
                newKL = log(p1 / p2) * p1
            KL += newKL

        missed1 = len(zeroCount1) + bothZero  # missed calcs are the missing
        missed2 = len(zeroCount2) + bothZero
        eps1 = (float(1.0) - totalMass1) / float(missed1)
        eps2 = (float(1.0) - totalMass2) / float(missed2)
        # print eps1
        # print eps2
        for missed in zeroCount1:  # i.e. where its a zero count in 1
            KL += (log(eps1 / missed) * eps1)
            totalMass1 += float(eps1)
            check += 1
            # totalMass2+=float(missed)
        for missed in zeroCount2:  # i.e. where its a zero count in 2
            KL += (log(missed / eps2) * missed)
            # totalMass1+=float(missed)
            check += 1
            totalMass2 += float(eps2)
        totalMass1 += (bothZero * eps1)
        totalMass2 += (bothZero * eps2)
        KL += (bothZero * (log(eps1 / eps2) * eps1))
        check += bothZero
        assert (check == self.vocab_size)
        assert abs(1.0 - totalMass1) <= 0.00000000001 and \
            abs(1.0 - totalMass2) <= 0.00000000001,\
            "NOT SUMMING TO 1. total mass 1 = " + str(totalMass1) + \
            " total mass 2 = " \
            + str(totalMass2) + str(contexttokens1) + str(contexttokens2)
        return KL

    def KL_divergence_very_fast_matrix(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions

        Parameters
        ----------
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
        Seeing if this speeds things up??
        We need to store an array for each bigram history
        linked to its target in v
        We can store entropy too doing this in an easy way.
        The problem is the initial run time for calcuating these.
        Given we've got sparse vectors
        we can speed this up as per the above methods.
        """
        p = numpy.asarray(p, dtype=numpy.float)
        q = numpy.asarray(q, dtype=numpy.float)

        return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))

    def store_top_N_matrices(self, n):
        # get the most frequent bigram histories
        # self.matrices = OrderedDict(bsr) #simple list of top
        # bigram histories indexed to a sparse matrix, their epsilon
        # and their entropy
        for bigram in self.bigrams.keys():
            self.ngram_denominator_map[bigram]
        i = 0
        for bigram, val in sorted(bigram.items(), key=itemgetter(1),
                                  reverse=True):  # most fertile bigrams
            self.cache[bigram] = self.entropy_continuation_fast(
                bigram.split(), returnEps=True,
                returnDist=True)
            # returns the epsilon and the sparse matrix
            i += 1
            if i == n:
                break

    def prob_distribution_over_completions(self, prefix):
        # assuming getting rid of the - prefix tries only way to do this
        prefix = prefix[:-1]
        probdist = defaultdict(float)
        Z = 0  # normalising constant
        missing = 0
        for key in self.unigrams.keys():
            if not prefix in key == None:
                missing += 1
                continue  # TODO what's the best way of smoothing?
            prob = self.ngram_prob([key], 1)  # unigram prob
            Z += prob
            probdist[key] = prob
            # total+=1
        print prefix + ":___\n"
        totalMass = 0
        for key, val in sorted(probdist.items(), key=itemgetter(1),
                               reverse=True):  # can get entropy on line for this?
            newprob = (1 / Z) * probdist[key]
            probdist[key] = newprob
            totalMass += newprob
        # print totalMass #should always sum to 1
        assert (abs(1.0 - totalMass) <= 0.00000000001), \
            "NOT SUMMING TO 1. total mass  = " + str(totalMass) + " " + prefix
        return probdist

    def raw_prob_completion(self, prefix, completion):
        # gives raw prob of a completion given its prefix, well,
        # smoothed because of sparse data
        # scale this number up to be more like a bigram
        if prefix == "<unk>" or self.prefixDict[prefix] == 0:
            # print "unseen"
            return 1.
        if prefix[-1] == "-":
            prefix = prefix[:-1]
        return float(self.prefixMap.get(self.glue_tokens(prefix, 1) +
                                        self.glue_tokens(completion, 1))) / self.prefixDict[prefix]

    def prob_fluent(self, ngram, order):
        """Gives argmax prob for a complete word finishing the partial word."""
        highest = 0.
        best = ""
        Z = 0.
        #epsilon = 1/len(self.unigrams.keys())
        count = 0
        for unigram in self.unigrams.keys():
            test = ngram[:-1] + [unigram]
            # print self.ngram_prob(test,order)
            # print self.raw_prob_completion(ngram[-1],unigram)
            # need a boosted log prob here, how do we do that for partial words
            completionprob = self.raw_prob_completion(ngram[-1], unigram)
            if completionprob == 0:
                prob = 0
            else:
                prob = self.raw_prob_completion(ngram[-1], unigram) \
                    * self.ngram_prob(test, order)
            # prob += epsilon # just to smooth,
            # we give everything the possibility of being
            # one maximum entropy value greater than 0
            if not prob == 0.0:
                count += 1
            Z += prob
            if prob > highest:
                highest = prob
                best = unigram
        print best
        return Z / float(count)

    def prob_fluent_simple(self, ngram, order):
        return self.ngram_prob(ngram, order) * 0.075
        # simply using the average drop in fluency from corpus estimate

    def repopulate_from_attrs(self, attrs, ngrams=None):
        """Loads a Kneser-Ney model"""

        def selective_update(m, k, v):
            if k in ngrams:
                m[k] = v

        def unrestricted_update(m, k, v):
            m[k] = v

        self.order = attrs['order']
        self.discount = attrs['discount']
        self.unigram_denominator = attrs['unigram_den']
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.ngram_non_zero_map = defaultdict(int)

        if ngrams is None:
            update = unrestricted_update
        else:
            update = selective_update

        flag = True
        db = open(attrs['num_db'], 'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_numerator_map, k, v)
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['den_db'], 'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_denominator_map, k, v)
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['nz_db'], 'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_non_zero_map, k, v)
            except EOFError:
                flag = False

        return self

#     # In the style of the old loading- is this needed?
#     @classmethod
#     def corpus(klass,attrs,ngrams=None):
#         """Loads a Kneser-Ney model"""
#
#         def selective_update(m,k,v):
#             if k in ngrams:
#                 m[k] = v
#         def unrestricted_update(m,k,v):
#             m[k] = v
#
#         self = klass() # inherits all methods/constructors
#
#         # loadable params
#         # Booleans and ints:
#         self.order = attrs['order']
#         self.partial_words = attrs['partial_words']
#         self.discount = attrs['discount']
#         self.verbose = attrs['partial_words']
#         self.train_length = attrs['train_length']
#         #gets initialized to check the training corpus length
#
#         self.bigram_types = attrs['bigram_types']
#         self.trigram_types = attrs['trigram_types']
#         self.vocab_size = attrs['vocab_size']
#         self.unigram_denominator = attrs['unigram_den']
#         # for unigrams we don't have contexts
#
#         # Stuff that needs to be loaded from clever pickling read in
#         self.ngram_numerator_map = defaultdict(int)
#         self.ngram_denominator_map = defaultdict(int)
#         self.ngram_non_zero_map = defaultdict(int)
#         self.unigram_contexts = defaultdict(list)
#         self.bigram_contexts = defaultdict(list)
#
#         self.unigrams = []
#         # Just an ordered list of unigrams that are followers (i.e. not <s>)
#         self.unigram_counts = defaultdict(int)
#         self.bigrams = []
#         # Essentially just a subset of ngram_denom for the bigram types,
#         # but not glued tokens :)
#         self.bigram_counts = defaultdict(int)
#
#         self.bigram_history_entropies = defaultdict(float)
#         self.trigram_history_entropies = defaultdict(float)
#
#
#         # Original methods
#         #self.order = attrs['order']
#         #self.discount = attrs['discount']
#         #self.unigram_denominator = attrs['unigram_den']
#         #self.ngram_numerator_map = defaultdict(int)
#         #self.ngram_denominator_map = defaultdict(int)
#         #self.ngram_non_zero_map = defaultdict(int)
#
#         if ngrams is None:
#             update = unrestricted_update
#         else:
#             update = selective_update
#
#         flag = True
#         db = open(attrs['num_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.ngram_numerator_map,k,v)
#             except EOFError:
#                 flag = False
#
#         flag = True
#         db = open(attrs['den_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.ngram_denominator_map,k,v)
#             except EOFError:
#                 flag = False
#
#         flag = True
#         db = open(attrs['nz_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.ngram_non_zero_map,k,v)
#             except EOFError:
#                 flag = False
#
#         flag = True
#         db = open(attrs['unigram_contexts_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.unigram_contexts,k,v)
#             except EOFError:
#                 flag = False
#
#         flag = True
#         db = open(attrs['bigram_contexts_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.bigram_contexts,k,v)
#             except EOFError:
#                 flag = False
#
#                 flag = True
#
#         db = open(attrs['unigram_counts_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.unigram_counts,k,v)
#             except EOFError:
#                 flag = False
#
#         flag = True
#         db = open(attrs['bigram_counts_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.bigram_counts,k,v)
#             except EOFError:
#                 flag = False
#
#         flag = True
#         db = open(attrs['bigram_history_entropies_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.bigram_history_entropies,k,v)
#             except EOFError:
#                 flag = False
#
#         flag = True
#         db = open(attrs['trigram_history_entropies_db'],'r')
#         while flag:
#             try:
#                 k = cPickle.corpus(db)
#                 v = cPickle.corpus(db)
#                 update(self.trigram_history_entropies,k,v)
#             except EOFError:
#                 flag = False
#
#         #TODO lists bigrams and unigrams, derivable from counts
#
#         return self

    def load(self, index_filename):
        """"Loads a Kneser-Ney model from index_filename"""
        f = safe_open(index_filename, 'rb')
        attributes = cPickle.load(f)
        self.order = attributes['order']
        self.partial_words = attributes['partial_words']
        self.discount = attributes['discount']
        self.verbose = attributes['verbose']
        self.bigram_types = attributes['bigram_types']
        self.trigram_types = attributes['trigram_types']
        self.vocab_size = attributes['vocab_size']
        self.unigram_denominator = attributes['unigram_denominator']
        self.ngram_numerator_map = attributes['ngram_numerator_map']
        self.ngram_denominator_map = attributes['ngram_denominator_map']
        self.ngram_non_zero_map = attributes['ngram_non_zero_map']
        self.unigram_contexts = attributes['unigram_contexts']
        self.bigram_contexts = attributes['bigram_contexts']
        self.unigram_counts = attributes['unigram_counts']
        self.bigram_counts = attributes['bigram_counts']
        # self.bigram_history_entropies = \
        # attributes['bigram_history_entropies'] # TODO not entropy cacheing
        # self.trigram_history_entropies = \
        # attributes['trigram_history_entropies']
        self.unigrams = attributes['unigrams']
        self.bigrams = attributes['bigrams']
        flush_and_close(f)

    def save(self, index_filename):
        """Saves a Kneser-Ney model to index_filename"""

        f = safe_open(index_filename, 'wb')
        cPickle.dump({'smoothing': 'kneser-ney',
                      'order': self.order, 'partial_words': self.partial_words,
                      'discount': self.discount, 'verbose': self.verbose,
                      'bigram_types': self.bigram_types,
                      'trigram_types': self.trigram_types,
                      'vocab_size': self.vocab_size,
                      'unigram_denominator': self.unigram_denominator,
                      'ngram_numerator_map': self.ngram_numerator_map,
                      'ngram_denominator_map': self.ngram_denominator_map,
                      'ngram_non_zero_map': self.ngram_non_zero_map,
                      'unigram_contexts': self.unigram_contexts,
                      'bigram_contexts': self.bigram_contexts,
                      'unigram_counts': self.unigram_counts,
                      'bigram_counts': self.bigram_counts,
                      'unigrams': self.unigrams, 'bigrams': self.bigrams}, f)
        flush_and_close(f)

#
#         #in the style of the old code- is this needed???
#         f = safe_open(index_filename,'wb')
#         num_db = os.path.abspath(index_filename) + '.num.db'
#         den_db = os.path.abspath(index_filename) + '.den.db'
#         nz_db = os.path.abspath(index_filename) + '.nz.db'
#         unigram_contexts_db = os.path.abspath(index_filename) +\
#            '.unigram_contexts.db'
#         bigram_contexts_db = os.path.abspath(index_filename) +\
#        ' '.bigram_contexts.db'
#         unigram_counts_db = os.path.abspath(index_filename) + \
#         '.unigram_counts.db'
#         bigram_counts_db = os.path.abspath(index_filename) + \
#        '.bigram_counts.db'
#         bigram_history_entropies_db = os.path.abspath(index_filename) +\
#         '.bigram_history_entropies.db'
#         trigram_history_entropies_db = os.path.abspath(index_filename) + \
#        '.trigram_history_entropies.db'
#         unigrams = os.path.abspath(index_filename) + '.unigrams.db'
#         bigrams = os.path.abspath(index_filename) + '.bigrams.db'
#
#
#         db = open(num_db,'w')
#         for k,v in self.ngram_numerator_map.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(den_db,'w')
#         for k,v in self.ngram_denominator_map.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(nz_db,'w')
#         for k,v in self.ngram_non_zero_map.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(unigram_contexts_db,'w')
#         for k,v in self.unigram_contexts.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(bigram_contexts_db,'w')
#         for k,v in self.bigram_contexts.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(unigram_counts_db,'w')
#         for k,v in self.unigram_counts.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(bigram_counts_db,'w')
#         for k,v in self.bigram_counts.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(bigram_history_entropies_db,'w')
#         for k,v in self.bigram_history_entropies.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#         db = open(trigram_history_entropies_db,'w')
#         for k,v in self.trigram_history_entropies.iteritems():
#             cPickle.dump(k,db,-1)
#             cPickle.dump(v,db,-1)
#         db.close()
#
#
#         #save the other stuff along with it
#         json.dump({'smoothing' : 'kneser-ney',\
#                      'order' : self.order, 'partial_words':
#                        self.partial_words, 'discount' : self.discount,
#                        'verbose' : self.verbose,\
#                      'bigram_types' : self.bigram_types, \
#                        'trigram_types' : self.trigram_types, \
#                        'vocab_size' : self.vocab_size, \
#                        'unigram_den' : self.unigram_denominator,\
#                         'num_db' : num_db, 'den_db' : den_db, \
#                        'nz_db' : nz_db, \
#                        'unigram_contexts_db': unigram_contexts_db,\
#                        'bigram_contexts_db': bigram_contexts_db,\
#                     'unigram_counts_db' : unigram_counts_db, \
#                            'bigram_counts_db': bigram_counts_db, \
#                'bigram_history_entropies_db' : bigram_history_entropies_db,\
#                     'trigram_histories_db' : trigram_history_entropies_db, \
#                    'unigrams_db': unigrams, 'bigrams_db': bigrams},f)
#         flush_and_close(f)
