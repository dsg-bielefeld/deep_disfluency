
from __future__ import division
# Script to add STIR (Hough and Purver 2014 EMNLP) Language Model features tos
# the input from ASR results/simulated ASR results
# this should work in an OOP style manner in that an object should be
# queried for its LM features incrementally as new words/POS's are added
# generalizes to a tagger setting more easily

import sys
import os
import numpy as np
import argparse
import pickle
import math
from copy import deepcopy

from deep_disfluency.language_model.ngram_language_model import KneserNeySmoothingModel
from deep_disfluency.language_model.ngram_language_model import NgramGraph
from feature_utils import get_diff_and_new_prefix, simulate_increco_data
from feature_utils import load_data_from_corpus_file


def save_vector_to_pickle(features, pkl_file):
    """Takes a list of lists and turns it into a np array to save."""
    v = np.asarray(features)
    with open(pkl_file, 'wb') as fp:
        pickle.dump(v, fp)


def swbd_folds_disfluency_corpus(input, num_files=496, num_folds=10):
    """Returns num_folds fold division of the input swbd PTB
    disfluency corpus in num_folds strings
    of the configuration of the division and the folds themselves.

    Keyword Arguments:
    input -- the (ID,.. up to n features) tuple which will be divded
    into num_fold tuples
    of the same type
    divs -- list(int), the size of each fold
    """
    # we have 10 divisions of 496 (?) files, the smallheldout corpus
    # already there so 9 to start with in rest
    folds = []
    config = []
    total = 0
    index = 0
    input = open(input)
    folds = []  # split main clean corpus into 9, have heldout as the other one

    # calculate the dividing points based on nearest int
    divs = []
    for i in range(1, num_folds):  # goes up to n-1th fold to get the split point
        split_point = int((i / num_folds) * num_files)
        divs.append(split_point)
    divs.append(num_files - 1)  # add the last one

    line = input.readline()  # first line
    for d in divs:
        subcorpus = ""
        posSubcorpus = ""
        targetstop = d
        currentSection = line.split(",")[0].split(":")[0]
        current = currentSection
        ranges = []
        print currentSection
        while index <= targetstop:
            ranges.append(current)
            while current == currentSection:
                subcorpus += line.split(",")[1] + "\n"
                posSubcorpus += input.readline().split(",")[1] + "\n"

                line = input.readline()  # read the next text level
                if not line:
                    break  # end of file, break to increment index?
                current = line.split(",")[0].split(":")[0]
            currentSection = current
            index += 1
        folds.append((tuple(ranges), subcorpus, posSubcorpus))
        # fold always has structure (ranges,wordsubcorpus(big string),
        # posSubcorpus(big string))

    print "no of folds = ", str(len(folds))
    for i in range(0, len(folds)):
        test = i
        if i == len(folds) - 1:
            heldout = i - 1
        else:
            heldout = i + 1
        training = []
        for index in range(0, len(folds)):
            if not index == heldout and not index == test:
                training.append(index)  # just appends an index
        config.append((tuple(training), heldout, test))
    print "config size", str(len(config))
    input.close()
    return config, folds


class IncrementalLanguageModelFeatureExtractor():
    """An object which consumes new prefixes of words and POS
    values and gives back a set of features as a vector each
    time its queried. Uses word graph objects to do so.
    """

    def __init__(self, lm, pos_model=None, edit_model=None, features=None,
                 back_search=8):
        self.lm = lm  # word language model
        self.pos_model = pos_model
        self.edit_model = edit_model
        self.features = features
        self.word_graph = NgramGraph(lm)
        self.pos_graph = NgramGraph(pos_model) if pos_model else None
        self.back_search = 8  # how far to go back

    def reset_graphs(self):
        self.word_graph = NgramGraph(self.lm)
        if self.pos_graph:
            self.pos_graph = NgramGraph(self.pos_model)

    def new_features_from_new_prefix(self, new_word_prefix, new_pos_prefix=None,
                                     backtrack=0,
                                     num_features=37):
        """Consumes new prefix of words (and POS tags if applicable)
        and returns the feature values in the order specified in self.features
        """
        results = []  # 2d will be len(new_word_prefix) * len(features)
        # first do the backtracking in the graphs
        self.word_graph = self.word_graph.subgraph([0, len(self.pos_graph.
                                                           word_graph)
                                                    - backtrack], self.lm)
        if self.pos_graph:
            self.pos_graph = self.pos_graph.subgraph([0, len(self.pos_graph.
                                                             word_graph)
                                                      - backtrack], self.lm)

        delta = self.lm.order - 1
        POSdelta = self.pos_model.order - 1
        # then go through the new prefix
        for i in range(0, len(new_word_prefix)):
            print new_word_prefix[i]
            if new_word_prefix[i] == "<laughter/>":
                # just add default values via copying or 0's
                if len(results) > 0:
                    results.append(deepcopy(results[-1]))
                else:
                    results.append(num_features * [0])
                continue

            self.word_graph.append(new_word_prefix[i], self.lm)
            # the uncleaned repair trigram/bigram
            rps_ngram = self.word_graph.word_graph[-self.lm.order:]
            # local boost of the trigram
            originalLocalWordWML = \
                self.lm.logprob_weighted_by_inverse_unigram_logprob(rps_ngram)
            originalLocalWordLogProb = \
                self.lm.logprob_weighted_by_sequence_length(rps_ngram)
            originalWordEntropy = \
                self.lm.entropy_continuation_very_fast(
                    rps_ngram[-(self.lm.order - 1):], self.lm.order)
            # overall boost of grammaticaility for utterance
            originalWordWML = \
                self.word_graph.logprob_weighted_by_inverse_unigram_logprob()
            # print "original = " + str(nonEditPrefixWords)
            originalWordLogProb = \
                self.word_graph.logprob_weighted_by_sequence_length()

            # Get features for this word
            # First edit model
            editngram = self.word_graph.word_graph[-self.edit_model.order:]
            editlogprob = sum(self.edit_model.tokens_logprob(editngram,
                                                             self.edit_model.order))
            test1 = editngram  # these good for midsentence ones...
            test2 = editngram
            test1 = list(["<s>"] + [editngram[-1]])
            edit1wordlogprob = sum(self.edit_model.tokens_logprob(test1,
                                                                  self.edit_model.order))
            edit2wordlogprob = sum(self.edit_model.tokens_logprob(test2,
                                                                  self.edit_model.order))
            editWML = self.edit_model.\
                logprob_weighted_by_inverse_unigram_logprob(editngram)
            #editdrop = 0
            # if len(self.word_graph.word_graph) >= self.edit_model.order:
            #    editprevious = self.word_graph.subgraph([0,len(self.\
            #                                word_graph.word_graph)-1],
            #                                            self.edit_model).\
            #                    logprob_weighted_by_inverse_unigram_logprob()
            #    editdrop = editprevious - editWML
            #editprevious = editWML

            # Second the normal LM
            # print wordngram #probs
            wordlogprob = self.lm.logprob_weighted_by_sequence_length(
                rps_ngram)
            wordWML = self.lm.logprob_weighted_by_inverse_unigram_logprob(
                rps_ngram)

            # or do we do this without the edit words i.e. previouswordngram =
            # nonEditPrefixWords[-lm.order:-1]; previous=
            # sum(lm.tokens_logprob(previouswordngram,lm.order))
            # wordprevious =
            #worddrop = wordprevious - wordWML
            #wordprevious = wordWML

            wordEntropy = self.lm.entropy_continuation_very_fast(
                rps_ngram[-delta:],
                self.lm.order)
            # wordEntropyPrevious =

            #wordEntropyHike = wordEntropy - wordEntropyPrevious
            # the ratio of surprisal of this event to the inherent entropy,
            # seems like a good measure
            #informationGain = math.log(-wordlogprob/wordEntropyPrevious,2)
            # expectedGainSquared = (wordWML - wordEntropyPrevious) * \
            #(wordWML - wordEntropyPrevious)
            #wordEntropyPrevious = wordEntropy

            results.append([editlogprob,
                            edit1wordlogprob,
                            edit2wordlogprob,
                            editWML,
                            wordlogprob, wordWML, wordEntropy])

            wordprevious = 0  # for WML drop
            if new_pos_prefix:
                self.pos_graph.append(new_pos_prefix[i], self.pos_model)
                rps_POS_ngram = \
                    self.pos_graph.word_graph[-self.pos_model.order:]
                # local boost of the trigram
                originalLocalPOSWML = \
                    self.pos_model.logprob_weighted_by_inverse_unigram_logprob(
                        rps_POS_ngram)
                originalLocalPOSLogProb = \
                    self.pos_model.logprob_weighted_by_sequence_length(
                        rps_POS_ngram)
                originalPOSEntropy = \
                    self.pos_model.entropy_continuation_very_fast(
                        rps_POS_ngram[-self.pos_model.order - 1:],
                        self.pos_model.order)
                # overall boost for utterance
                originalPOSWML = \
                    self.pos_graph.logprob_weighted_by_inverse_unigram_logprob()
                originalPOSLogProb = \
                    self.pos_graph.logprob_weighted_by_sequence_length()
                # now for POS, given we've done the words

                poslogprob = self.pos_model.\
                    logprob_weighted_by_sequence_length(rps_POS_ngram)
                posWML = self.pos_model.\
                    logprob_weighted_by_inverse_unigram_logprob(
                        rps_POS_ngram)

                # or do we do this without the edit words i.e. previouswordngram
                # = nonEditPrefixWords[-lm.order:-1]; previous=
                # sum(lm.tokens_logprob(previouswordngram,lm.order))
                # posprevious =
                #posdrop = posprevious - posWML
                #posprevious = posWML

                posEntropy = self.pos_model.entropy_continuation_very_fast(
                    rps_POS_ngram[-POSdelta:],
                    self.pos_model.order)
                # posEntropyPrevious =

                #posEntropyHike = posEntropy - posEntropyPrevious
                # posInformationGain = math.log(-poslogprob/posEntropyPrevious,\
                # 2)
                results.append([poslogprob, posWML, posEntropy])

            j = len(self.word_graph.word_graph) - 1
            start = j
            reparandumLength = 1
            k = -1
            POSprevious = 0
            while j >= max([j - self.back_search, self.lm.order - 1]):
                # backwards search

                rmwordngram = self.word_graph.word_graph[k - self.lm.order:k]
                # the cleaned rpwordngram (i.e. excised of reparnadum)
                rpwordngram = self.word_graph.word_graph[k - self.lm.order:k - 1] +\
                    [rps_ngram[-1]]
                if len(rpwordngram) < self.lm.order:  # no more..
                    break
                # print "rpwordngram"
                # print rpwordngram

                wordWML = self.lm.logprob_weighted_by_inverse_unigram_logprob(
                    rpwordngram)
                wordlogprob = self.lm.logprob_weighted_by_sequence_length(
                    rpwordngram)
                wordRMWML = self.lm.\
                    logprob_weighted_by_inverse_unigram_logprob(
                        rmwordngram)
                wordRMlogprob = self.lm.logprob_weighted_by_sequence_length(
                    rmwordngram)
                # uncleaned bigram/unigram around the rm0 boundary-
                # should be high
                wordRM0Entropy = self.lm.entropy_continuation_very_fast(
                    rmwordngram[-(
                        self.lm.order - 1):],
                    self.lm.order)
                wordEntropy = self.lm.entropy_continuation_very_fast(
                    rpwordngram[-(
                        self.lm.order - 1):],
                    self.lm.order)
                wordLocalEntropyReduce = wordEntropy - originalWordEntropy

                wordLocalWMLBoost = wordWML - originalLocalWordWML
                wordLocalProbBoost = wordlogprob - originalLocalWordLogProb
                # wordCleanTest = nonEditPrefixWords[:k-1] + \
                #[nonEditPrefixWords[-1]] #the last one

                wordCleanTest = self.word_graph.subgraph([0, k - 1], self.lm)
                wordCleanTest.extend(self.word_graph.subgraph([
                    len(self.word_graph.word_graph) - 1,
                    len(self.word_graph.word_graph)],
                    self.lm), self.lm)

                # print "word clean test"
                # print wordCleanTest
                # TODO this is where we need the graph methods, as
                # we can quickly compute these that have already been computed
                # wordWMLBoost = lm.\
                # logprob_weighted_by_inverse_unigram_logprob(wordCleanTest)
                #- originalWordWML
                # wordProbBoost = lm.logprob_weighted_by_sequence_length(
                # wordCleanTest) - originalWordLogProb
                wordWMLBoost = wordCleanTest.\
                    logprob_weighted_by_inverse_unigram_logprob() -\
                    originalWordWML
                wordProbBoost = wordCleanTest.\
                    logprob_weighted_by_sequence_length() -\
                    originalWordLogProb

                # or do we do this without the edit words
                #i.e. previouswordngram = nonEditPrefixWords[-lm.order:-1];
                #previous= sum(lm.tokens_logprob(previouswordngram,lm.order))
                if not j == start:
                    worddrop = wordWMLBoost - wordprevious
                else:
                    worddrop = 0
                wordprevious = wordWMLBoost

                # wordKL = lm.KL_divergence_continuation_fast(rmwordngram[1:],\
                # rpwordngram[1:]) #predictibility of next word
                # may have to leave this unfort..
                # TODO speed up and put back in
                #wordKL = 0.0

                # referenceFile.write(reference+","+str(i-delta)+","+\
                # str(j-delta)+",")
                # if len(repairOnsetNgram) < 3: test = ["null"] + \
                # list(repairOnsetNgram)
                # else: test = list(repairOnsetNgram)
                # referenceFile.write(str(test)[1:-1].replace("'","").\
                # replace(" ","")+",") #the words, only bigram context?
                # referenceFile.write(str(test)[1:-1].replace("'","").\
                # replace(" ","")+",")
                # just one for now, could add a partial word/prefix feature too
                # see if there's a repeat
                rm3rp3 = 0
                if rmwordngram[-1] == rmwordngram[-1]:
                    rm3rp3 = 1
                results[-1].extend([wordlogprob, wordWML, wordRMlogprob,
                                    wordRMWML,
                                    wordLocalProbBoost, wordLocalWMLBoost,
                                    wordProbBoost, wordWMLBoost, worddrop,
                                    wordEntropy, wordLocalEntropyReduce,
                                    wordRM0Entropy, rm3rp3])
                if self.pos_model:
                    rmPOSngram = self.pos_graph.word_graph[k -
                                                           self.pos_model.order:k]
                    rpPOSngram = self.pos_graph.word_graph[k -
                                                           self.pos_model.order:k - 1] +\
                        [rps_POS_ngram[-1]]
                    # print "rpPOSngram"
                    # print rpPOSngram

                    POSlogprob = self.pos_model.\
                        logprob_weighted_by_sequence_length(rpPOSngram)
                    POSWML = self.pos_model.\
                        logprob_weighted_by_inverse_unigram_logprob(rpPOSngram)
                    POSRMlogprob = \
                        self.pos_model.logprob_weighted_by_sequence_length(
                            rmPOSngram)
                    POSRMWML = self.pos_model.\
                        logprob_weighted_by_inverse_unigram_logprob(rmPOSngram)

                    POSLocalWMLBoost = POSWML - originalLocalPOSWML
                    POSLocalProbBoost = POSlogprob - originalLocalPOSLogProb

                    POSEntropy = self.pos_model.entropy_continuation_very_fast(
                        rpPOSngram[-(self.pos_model.order - 1):],
                        self.pos_model.order)  # clean
                    POSLocalEntropyReduce = POSEntropy - originalPOSEntropy
                    POSRM0Entropy = self.pos_model.\
                        entropy_continuation_very_fast(rmPOSngram[
                            -(self.pos_model.order - 1):],
                            self.pos_model.order)
                    # uncleaned bigram around the rm0 boundary- should be high
                    # POSCleanTest = nonEditPrefixPOS[:k-1] + \
                    #[nonEditPrefixPOS[-1]]
                    POSCleanTest = self.pos_graph.subgraph([0, k - 1],
                                                           self.pos_model)
                    POSCleanTest.extend(self.pos_graph.subgraph(
                        [len(self.pos_graph.word_graph) - 1,
                         len(self.pos_graph.word_graph)],
                        self.pos_model),
                        self.pos_model)

                    # print "POSCleanTest"
                    # print POSCleanTest
                    # POSWMLBoost = pos_model.\
                    # logprob_weighted_by_inverse_unigram_logprob(POSCleanTest)
                    # - originalPOSWML
                    # POSProbBoost = pos_model.\
                    # logprob_weighted_by_sequence_length(POSCleanTest) -\
                    # originalPOSLogProb

                    POSWMLBoost = POSCleanTest.\
                        logprob_weighted_by_inverse_unigram_logprob() - \
                        originalPOSWML
                    POSProbBoost = POSCleanTest.\
                        logprob_weighted_by_sequence_length() - \
                        originalPOSLogProb
                    # this should be positive if it's getting better or do we do
                    # this without the edit words
                    #i.e. previouswordngram = nonEditPrefixWords[-lm.order:-1];
                    # previous=
                    # sum(lm.tokens_logprob(previouswordngram,lm.order))
                    if not j == start:
                        POSdrop = POSWMLBoost - POSprevious
                    else:
                        POSdrop = 0
                    POSprevious = POSWMLBoost
                    POSKL = self.pos_model.KL_divergence_continuation_fast(
                        rmPOSngram[-(self.pos_model.order - 1):],
                        rpPOSngram[-(self.pos_model.order - 1):])
                    rm3rp3POS = 0
                    # see if there's a repeated POS
                    if rpPOSngram[-1] == rmPOSngram[-1]:
                        rm3rp3POS = 1
                    results[-1].extend([POSlogprob, POSWML, POSRMlogprob,
                                        POSRMWML, POSKL, POSLocalProbBoost,
                                        POSLocalWMLBoost, POSProbBoost,
                                        POSWMLBoost, POSdrop, POSEntropy,
                                        POSLocalEntropyReduce, POSRM0Entropy,
                                        rm3rp3POS])
                j -= 1
        print len(results), len(results[0])
        return results


def extract_lang_model_features_from_increco(data,
                                             lm,
                                             pos_model=None,
                                             edit_model=None,
                                             dialogues=None,
                                             reset_at_utterance=False):
    """Language model features
    for a given set of increco results (incremental updates) to
    utterances and outputs a list of the values of the same dimension
    could be reasonably large size, i.e. length * n_features
    """

    f_extractor = IncrementalLanguageModelFeatureExtractor(
        lm,
        pos_model=pos_model,
        edit_model=edit_model,
        features=[],
        back_search=8
    )

    dialogue_vector = []
    # f_extractor.reset_graphs()
    frames_data, lex_data, pos_data, indices_data, labels_data = data
    lex_data = deepcopy([[x[0][1]] for x in lex_data])  # simulating prefix
    pos_data = deepcopy([[x[1]] for x in pos_data])
    for word_prefix, pos_prefix, label in zip(lex_data, pos_data, labels_data):
        print "word_prefix", word_prefix
        print "pos_prefix", pos_prefix
        r = f_extractor.new_features_from_new_prefix(
            word_prefix,
            new_pos_prefix=pos_prefix,
            backtrack=0
        )
        dialogue_vector.append(deepcopy(r))
        if reset_at_utterance and "t/>" in label:
            f_extractor.reset_graphs()
    return dialogue_vector


def extract_language_model_string(clean_model_file,
                                  dialogue_ranges,
                                  pos_tags=True,
                                  utt_sep=True):
    """Returns newline-separated string for language model corpus.
    Optionally the pos tags as a parallel corpus string if pos_tags== True
    """
    corpus_string = ""
    pos_string = ""
    # 3389:A:0:qw,whos your favorite team
    # POS,WPBES PRP$ JJ NN
    file = open(clean_model_file)
    clean_lines = [line.strip('\n') for line in file]
    file.close()
    key = ""
    for line in clean_lines:
        cells = line.split(",")
        if not "POS" in line:
            key = cells[0].split(":")[0] + cells[0].split(":")[1]
            current_line = cells[1].strip("\n")
            continue
        else:
            current_pos_line = cells[1].strip("\n")
        if not key in dialogue_ranges:
            continue
        print "dialogue", key
        corpus_string += current_line + "\n"
        pos_string += current_pos_line + "\n"

        # print "dialogue",dialogue[0]
        #words = [ x[0][1] for x in dialogue[1][1] ]
        #pos = [ x[1] for x in dialogue[1][2] ]
        #labels = dialogue[1][4]
        # print labels

        # for w,p,l in zip(words,pos,labels):
        #    corpus_string+=w + " "
        ##    pos_string+=p + " "
        #    if "t/>" in l and utt_sep:
        ##       corpus_string = corpus_string.strip()+"\n"
        #        pos_string = pos_string.strip()+"\n"
    if pos_tags:
        return corpus_string, pos_string
    else:
        return corpus_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language model \
    training and addition of language model features from it.')

    parser.add_argument('-i', action='store', dest='corpusFile',
                        default='../data/disfluency_detection/\
                        switchboard/swbd_disf_train_1_partial_data.csv',
                        help='Location of the corpus to annotate with\
                        language model features.')
    parser.add_argument('-lm', action='store', dest='cleanModelDir',
                        default="../data/lm_corpora",
                        help='Location of where to write a clean language\
                        model files out of this corpus.')
    parser.add_argument(
        '-f', action='store',
        dest='featureFolder',
        default='../data/disfluency_detection/feature_matrices/lm_features',
        help="The location of the data vectors to be outputted.")
    parser.add_argument('-order', action='store', default=3, type=int)
    parser.add_argument('-xlm', action='store_true',
                        dest='crossValLanguageModelTraining',
                        default=False,
                        help='Whether to use a cross language model\
                        training to be used for getting lm features on\
                        the same data.')
    parser.add_argument('-p', action='store_true', dest='partial_words',
                        default=False,
                        help='Whether to use partial words or not.')
    parser.add_argument('-e', action='store_true', dest='edit_input',
                        default=False,
                        help='Whether to train an edit term model or not.')
    parser.add_argument('-u', action='store_true', dest='utt_sep',
                        default=False,
                        help='Whether to separate the utterances or not\
                        in training or testing.')
    args = parser.parse_args()
    discount = 0.7  # always 0.7 for now, works well
    second_corpus = None
    pos_second_corpus = None

    try:
        os.mkdir(args.vectorFolder)
    except Exception:
        print "couldn't create", args.vectorFolder, "might already be there"

    if args.edit_input:
        print "Training edit term Language Model..."
        # stays the same across folds
        edit_lm_corpus_file = open(args.cleanModelDir +
                                   "/swbd_disf_train_1_edit.text")
        edit_lines = [line.strip("\n").split(",")[1]
                      for line in edit_lm_corpus_file if not "POS,"
                      in line and not line.strip("\n") == ""]
        edit_split = int(0.9 * len(edit_lines))
        edit_lm_corpus = "\n".join(edit_lines[:edit_split])
        heldout_edit_lm_corpus = "\n".join(edit_lines[edit_split:])
        edit_lm = KneserNeySmoothingModel(train_corpus=edit_lm_corpus,
                                          heldout_corpus=heldout_edit_lm_corpus,
                                          order=2, discount=discount)

    dialogues = sorted(load_data_from_corpus_file(args.corpusFile))
    num_folds = 10
    fold_size = int(len(dialogues) * 1 / num_folds)  # 10 fold cross lm
    print "fold_size", fold_size
    folds = {}
    lm_corpus = {}  # lm corpus always
    pos_lm_corpus = {}  # pos tags corpus

    # 1. From the dialogues get the language model strings in the fold
    # and the ranges for the output file/vectors
    previous_split = 0
    split = fold_size
    for f in range(1, num_folds + 1):
        # print lm_string

        fold_ranges = [x[0] for x in dialogues[previous_split: split]]
        folds[f] = fold_ranges
        print "current range", previous_split, split
        lm_string, pos_string = extract_language_model_string(
            args.cleanModelDir + "/swbd_disf_train_1_clean.text",
            fold_ranges,
            pos_tags=True,
            utt_sep=args.utt_sep)
        lm_corpus[f] = lm_string
        pos_lm_corpus[f] = pos_string
        print "number of strings", len(lm_string.split("\n"))
        split += fold_size
        previous_split += fold_size

    # 2. Cross val model for the vector values
    if args.crossValLanguageModelTraining:
        # if using the cross val LM training method:
        for config_key in sorted(folds.keys()):
            a_corpus = ""
            a_pos_corpus = ""
            a_heldout_corpus = ""
            a_pos_heldout_corpus = ""
            heldout_key = [x for x in sorted(folds.keys())
                           if x != config_key][-1]
            for lm_key in lm_corpus.keys():
                if lm_key == config_key or lm_key == heldout_key:
                    continue
                a_corpus += lm_corpus[lm_key] + '\n'
                a_pos_corpus += pos_lm_corpus[lm_key] + "\n"
            a_heldout_corpus = lm_corpus[heldout_key]
            a_pos_heldout_corpus = pos_lm_corpus[heldout_key]
            print "training sub lm"
            print a_corpus
            sublm = KneserNeySmoothingModel(order=args.order,
                                            discount=discount,
                                            partial_words=args.partial_words,
                                            train_corpus=a_corpus,
                                            heldout_corpus=a_heldout_corpus,
                                            second_corpus=second_corpus)
            subposlm = KneserNeySmoothingModel(order=args.order,
                                               discount=discount,
                                               partial_words=args.partial_words,
                                               train_corpus=a_pos_corpus,
                                               heldout_corpus=a_pos_heldout_corpus,
                                               second_corpus=pos_second_corpus)
            for d in dialogues:
                speaker_id, data = d
                if not speaker_id in folds[f]:
                    continue
                print "processing", speaker_id
                dialogue_features = extract_lang_model_features_from_increco(
                    data,
                    sublm,
                    pos_model=subposlm,
                    edit_model=edit_lm,
                    reset_at_utterance=True)
                print "saving", speaker_id
                save_vector_to_pickle(dialogue_features,
                                      args.vectorFolder + "/" + speaker_id + ".pkl")
