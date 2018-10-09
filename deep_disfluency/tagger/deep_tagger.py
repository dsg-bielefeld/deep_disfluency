from __future__ import division
import numpy as np
import cPickle
import os
from copy import deepcopy
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import gensim
from nltk.tag import CRFTagger
import re

from deep_disfluency.language_model.ngram_language_model \
    import KneserNeySmoothingModel
from deep_disfluency.utils.tools \
    import convert_from_inc_disfluency_tags_to_eval_tags
from deep_disfluency.utils.tools import \
    verify_dialogue_data_matrices_from_folder
from deep_disfluency.utils.tools import \
    dialogue_data_and_indices_from_matrix
from deep_disfluency.load.load import load_tags
from deep_disfluency.rnn.elman import Elman
from deep_disfluency.rnn.lstm import LSTM
from deep_disfluency.rnn.test_if_using_gpu import test_if_using_GPU
from deep_disfluency.decoder.hmm import FirstOrderHMM
from deep_disfluency.decoder.noisy_channel import SourceModel
from deep_disfluency.embeddings.load_embeddings import populate_embeddings
from deep_disfluency.feature_extraction.feature_utils import \
    load_data_from_disfluency_corpus_file
from deep_disfluency.evaluation.disf_evaluation import \
    get_tag_data_from_corpus_file
from utils import process_arguments, get_last_n_features
from deep_disfluency.feature_extraction.feature_utils import \
    sort_into_dialogue_speakers


class IncrementalTagger(object):
    """A generic incremental tagging object which can deal with incremental
    word-by-word (and revokable) input. It doesn't do anything.
    """

    def __init__(self, config_file, config_number, saved_model_folder=None):
        print "Initializing Tagger"
        # initialize (word, pos) tuple graph
        self.window_size = 0
        self.word_graph = []
        self.output_tags = []
        self.timing_model = None
        self.timing_model_scaler = None

    def load_model_from_folder(self, saved_model_folder, model_type):
        """Assumes some machine learning model needs to loaded.
        """
        raise NotImplementedError

    def tag_new_word(self, word, pos=None, timing=None, rollback=0):
        """Tags current word and adds to output tags.
        """
        self.rollback(rollback)

    def rollback(self, backwards):
        """Revoke the right frontier of the input and labels back backwards.
        """
        self.word_graph = self.word_graph[:len(self.word_graph) - backwards]
        self.output_tags = self.output_tags[:len(self.output_tags) - backwards]

    def tag_new_prefix(self, prefix, rollback=0):
        self.rollback(rollback)
        for w, p, t in prefix:
            self.tag_new_word(w, p, t)

    def reset(self):
        self.output_tags = []


class DeepDisfluencyTagger(IncrementalTagger):
    """A deep-learning driven incremental disfluency tagger
    (and optionally utterance-segmenter).

    Tags each word with the following:
    <f/> - a fluent word
    <e/> - an edit term word, not necessarily inside a repair structure
    <rms id="N"/> - reparandum start word for repair with ID number N
    <rm id="N"/> - mid-reparandum word for repair N
    <i id="N"/> - interregnum word for repair N
    <rps id="N"/> - repair onset word for repair N
    <rp id="N"/> - mid-repair word for repair N
    <rpn id="N"/> - repair end word for substitution or repetition repair N
    <rpnDel id="N"/> - repair end word for a delete repair N

    If in joint utterance segmentation mode
    according to the config file,
    the following utterance segmentation tags are used:

    <cc/> - a word which continues the current utterance and whose
            following word will continue it
    <ct/> - a word which continues the current utterance and is the
            last word of it
    <tc/> - a word which is the beginning of an utterance and whose following
            word will continue it
    <tt/> - a word constituting an entire utterance
    """

    def __init__(self, config_file=None,
                 config_number=None,
                 saved_model_dir=None,
                 pos_tagger=None,
                 language_model=None,
                 pos_language_model=None,
                 edit_language_model=None,
                 timer=None,
                 timer_scaler=None,
                 use_timing_data=False,
                 use_decoder=True):

        if not config_file:
            config_file = "experiments/experiment_configs.csv"
            config_number = 35
            print "No config file, using default", config_file, config_number

        config_file = os.path.join(
            os.path.dirname(__file__), '..', config_file)
        saved_model_dir = os.path.join(
            os.path.dirname(__file__), '..', saved_model_dir)

        super(DeepDisfluencyTagger, self).__init__(config_file,
                                                   config_number,
                                                   saved_model_dir)
        print "Processing args from config number {} ...".format(config_number)
        self.args = process_arguments(config_file,
                                      config_number,
                                      use_saved=False,
                                      hmm=True)
        #  separate manual setting
        setattr(self.args, "use_timing_data", use_timing_data)
        print "Intializing model from args..."
        self.model = self.init_model_from_config(self.args)

        # load a model from a folder if specified
        if saved_model_dir:
            print "Loading saved weights from", saved_model_dir
            self.load_model_params_from_folder(saved_model_dir,
                                               self.args.model_type)
        else:
            print "WARNING no saved model params, needs training."
            print "Loading original embeddings"
            self.load_embeddings(self.args.embeddings)

        if pos_tagger:
            print "Loading POS tagger..."
            self.pos_tagger = pos_tagger
        elif self.args.pos:
            print "No POS tagger specified,loading default CRF switchboard one"
            self.pos_tagger = CRFTagger()
            tagger_path = os.path.dirname(os.path.realpath(__file__)) +\
                "/../feature_extraction/crfpostagger"
            self.pos_tagger.set_model_file(tagger_path)

        if self.args.n_language_model_features > 0 or \
                'noisy_channel' in self.args.decoder_type:
            print "training language model..."
            self.init_language_models(language_model,
                                      pos_language_model,
                                      edit_language_model)

        self.timing_model = None
        self.timing_model_scaler = None
        if timer:
            print "loading timer..."
            self.timing_model = timer
            self.timing_model_scaler = timer_scaler
        elif self.args.use_timing_data:
            # self.timing_model = None
            # self.timing_model_scaler = None
            print "No timer specified, using default switchboard one"
            timer_path = os.path.dirname(os.path.realpath(__file__)) +\
                '/../decoder/timing_models/' + \
                'LogReg_balanced_timing_classifier.pkl'
            with open(timer_path, 'rb') as fid:
                self.timing_model = cPickle.load(fid)
            timer_scaler_path = os.path.dirname(os.path.realpath(__file__)) +\
                '/../decoder/timing_models/' + \
                'LogReg_balanced_timing_scaler.pkl'
            with open(timer_scaler_path, 'rb') as fid:
                self.timing_model_scaler = cPickle.load(fid)
                # TODO a hack
                # self.timing_model_scaler.scale_ = \
                #    self.timing_model_scaler.std_.copy()
        else:
            print "Not using timing data"

        print "Loading decoder..."
        self.hmm_dict = deepcopy(self.tag_to_index_map)
        # add the interegnum tag
        if "disf" in self.args.tags:
            intereg_ind = len(self.hmm_dict.keys())
            interreg_tag = \
                "<i/><cc/>" if "uttseg" in self.args.tags else "<i/>"
            # add the interregnum tag
            self.hmm_dict[interreg_tag] = intereg_ind

        # decoder_file = os.path.dirname(os.path.realpath(__file__)) + \
        #     "/../decoder/model/{}_tags".format(self.args.tags)
        noisy_channel = None
        if 'noisy_channel' in self.args.decoder_type:
            noisy_channel = SourceModel(self.lm, self.pos_lm,
                                        uttseg=self.args.do_utt_segmentation)
        self.decoder = None
        if use_decoder:
            self.decoder = FirstOrderHMM(
                self.hmm_dict,
                markov_model_file=self.args.tags,
                timing_model=self.timing_model,
                timing_model_scaler=self.timing_model_scaler,
                constraint_only=True,
                noisy_channel=noisy_channel
            )

        # getting the states in the right shape
        self.state_history = []
        self.softmax_history = []
        # self.convert_to_output_tags = get_conversion_method(self.args.tags)
        self.reset()

    def init_language_models(self, language_model=None,
                             pos_language_model=None,
                             edit_language_model=None):
        clean_model_dir = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/lm_corpora"
        if language_model:
            self.lm = language_model
        else:
            print "No language model specified, using default switchboard one"
            lm_corpus_file = open(clean_model_dir +
                                  "/swbd_disf_train_1_clean.text")
            lines = [line.strip("\n").split(",")[1] for line in lm_corpus_file
                     if "POS," not in line and not line.strip("\n") == ""]
            split = int(0.9 * len(lines))
            lm_corpus = "\n".join(lines[:split])
            heldout_lm_corpus = "\n".join(lines[split:])
            lm_corpus_file.close()
            self.lm = KneserNeySmoothingModel(
                order=3,
                discount=0.7,
                partial_words=self.args.partial_words,
                train_corpus=lm_corpus,
                heldout_corpus=heldout_lm_corpus,
                second_corpus=None)
        if pos_language_model:
            self.pos_lm = pos_language_model
        elif self.args.pos:
            print "No pos language model specified, \
            using default switchboard one"
            lm_corpus_file = open(clean_model_dir +
                                  "/swbd_disf_train_1_clean.text")
            lines = [line.strip("\n").split(",")[1] for line in lm_corpus_file
                     if "POS," in line and not line.strip("\n") == ""]
            split = int(0.9 * len(lines))
            lm_corpus = "\n".join(lines[:split])
            heldout_lm_corpus = "\n".join(lines[split:])
            lm_corpus_file.close()
            self.pos_lm = KneserNeySmoothingModel(
                order=3,
                discount=0.7,
                partial_words=self.args.partial_words,
                train_corpus=lm_corpus,
                heldout_corpus=heldout_lm_corpus,
                second_corpus=None)
        if edit_language_model:
            self.edit_lm = edit_language_model
        else:
            edit_lm_corpus_file = open(clean_model_dir +
                                       "/swbd_disf_train_1_edit.text")
            edit_lines = [line.strip("\n").split(",")[1]
                          for line in edit_lm_corpus_file
                          if "POS," not in line and not line.strip("\n") == ""]
            edit_split = int(0.9 * len(edit_lines))
            edit_lm_corpus = "\n".join(edit_lines[:edit_split])
            heldout_edit_lm_corpus = "\n".join(edit_lines[edit_split:])
            edit_lm_corpus_file.close()
            self.edit_lm = KneserNeySmoothingModel(
                train_corpus=edit_lm_corpus,
                heldout_corpus=heldout_edit_lm_corpus,
                order=2,
                discount=0.7)
            # TODO an object for getting the lm features incrementally
            # in the language model

    def init_model_from_config(self, args):
        # for feat, val in args._get_kwargs():
        #     print feat, val, type(val)
        if not test_if_using_GPU():
            print "Warning: not using GPU, might be a bit slow"
            print "\tAdjust Theano config file ($HOME/.theanorc)"
        print "loading tag to index maps..."
        label_path = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/tag_representations/{}_tags.csv".format(args.tags)
        word_path = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/tag_representations/{}.csv".format(args.word_rep)
        pos_path = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/tag_representations/{}.csv".format(args.pos_rep)
        self.tag_to_index_map = load_tags(label_path)
        self.word_to_index_map = load_tags(word_path)
        self.pos_to_index_map = load_tags(pos_path)
        self.model_type = args.model_type
        vocab_size = len(self.word_to_index_map.keys())
        emb_dimension = args.emb_dimension
        n_hidden = args.n_hidden
        n_extra = args.n_language_model_features + args.n_acoustic_features
        n_classes = len(self.tag_to_index_map.keys())
        self.window_size = args.window
        n_pos = len(self.pos_to_index_map.keys())
        update_embeddings = args.update_embeddings
        lr = args.lr
        print "Initializing model of type", self.model_type, "..."
        if self.model_type == 'elman':
            model = Elman(ne=vocab_size,
                          de=emb_dimension,
                          nh=n_hidden,
                          na=n_extra,
                          n_out=n_classes,
                          cs=self.window_size,
                          npos=n_pos,
                          update_embeddings=update_embeddings)
            self.initial_h0_state = model.h0.get_value()
            self.initial_c0_state = None

        elif self.model_type == 'lstm':
            model = LSTM(ne=vocab_size,
                         de=emb_dimension,
                         n_lstm=n_hidden,
                         na=n_extra,
                         n_out=n_classes,
                         cs=self.window_size,
                         npos=n_pos,
                         lr=lr,
                         single_output=True,
                         cost_function='nll')
            self.initial_h0_state = model.h0.get_value()
            self.initial_c0_state = model.c0.get_value()
        else:
            raise NotImplementedError('No model init for {0}'.format(
                self.model_type))
        return model

    def load_model_params_from_folder(self, model_folder, model_type):
        if model_type in ["lstm", "elman"]:
            self.model.load_weights_from_folder(model_folder)
            self.initial_h0_state = self.model.h0.get_value()
            if model_type == "lstm":
                self.initial_c0_state = self.model.c0.get_value()
        else:
            raise NotImplementedError('No weight loading for {0}'.format(
                model_type))

    def load_embeddings(self, embeddings_name):
        # load pre-trained embeddings
        embeddings_dir = os.path.dirname(os.path.realpath(__file__)) +\
            "/../embeddings/"
        pretrained = gensim.models.Word2Vec.load(embeddings_dir +
                                                 embeddings_name)
        print "emb shape", pretrained[pretrained.index2word[0]].shape
        # print pretrained[0].shape
        # assign and fill in the gaps
        emb = populate_embeddings(self.args.emb_dimension,
                                  len(self.word_to_index_map.items()),
                                  self.word_to_index_map,
                                  pretrained)
        self.model.load_weights(emb=emb)

    def standardize_word_and_pos(self, word, pos=None,
                                 proper_name_pos_tags=["NNP", "NNPS",
                                                       "CD", "LS",
                                                       "SYM", "FW"]):
        word = word.lower().replace("'", "")  # no punctuation
        if not pos and self.pos_tagger:
            pos = self.pos_tagger.tag([])  # TODO
        if pos:
            pos = pos.upper()
            if pos in proper_name_pos_tags and "$unc$" not in word:
                word = "$unc$" + word
            if self.pos_to_index_map.get(pos) is None:
                # print "unknown pos", pos
                pos = "<unk>"
        if self.word_to_index_map.get(word) is None:
            # print "unknown word", word
            word = "<unk>"
        return word, pos

    def tag_new_word(self, word, pos=None, timing=None, extra=None,
                     diff_only=True, rollback=0):
        """Tag new incoming word and update the word and tag graphs.

        :param word: the word to consume/tag
        :param pos: the POS tag to consume/tag (optional)
        :param timing: the duration of the word (optional)
        :param diff_only: whether to output only the diffed suffix,
        if False, outputs entire output tags
        :param rollback: the number of words to rollback
        in the case of changed word hypotheses from an ASR
        """
        self.rollback(rollback)
        if pos is None and self.args.pos:
            # if no pos tag provided but there is a pos-tagger, tag word
            test_words = [unicode(x) for x in
                          get_last_n_features(
                "words",
                self.word_graph,
                len(self.word_graph) - 1,
                n=4
            )
            ] + [unicode(word.lower())]
            pos = self.pos_tagger.tag(test_words)[-1][1]
            # print "tagging", word, "as", pos
        # 0. Add new word to word graph
        word, pos = self.standardize_word_and_pos(word, pos)
        # print "New word:", word, pos
        self.word_graph.append((word, pos, timing))
        # 1. load the saved internal rnn state
        # TODO these nets aren't (necessarily) trained statefully
        # The internal state in training self.args.bs words back
        # are the inital ones in training, however here
        # They are the actual state reached.
        if self.state_history == []:
            c0_state = self.initial_c0_state
            h0_state = self.initial_h0_state
        else:
            if self.model_type == "lstm":
                c0_state = self.state_history[-1][0][-1]
                h0_state = self.state_history[-1][1][-1]
            elif self.model_type == "elman":
                h0_state = self.state_history[-1][-1]

        if self.model_type == "lstm":
            self.model.load_weights(c0=c0_state,
                                    h0=h0_state)
        elif self.model_type == "elman":
            self.model.load_weights(h0=h0_state)
        else:
            raise NotImplementedError("no history loading for\
                             {0} model".format(self.model_type))

        # 2. do the softmax output with converted inputs
        word_window = [self.word_to_index_map[x] for x in
                       get_last_n_features("words", self.word_graph,
                                           len(self.word_graph) - 1,
                                           n=self.window_size)
                       ]
        pos_window = [self.pos_to_index_map[x] for x in
                      get_last_n_features("POS", self.word_graph,
                                          len(self.word_graph) - 1,
                                          n=self.window_size)
                      ]
        # print "word_window, pos_window", word_window, pos_window
        if self.model_type == "lstm":
            h_t, c_t, s_t = self.model.\
                soft_max_return_hidden_layer([word_window], [pos_window])
            self.softmax_history.append(s_t)
            if len(self.state_history) == 20:  # just saving history
                self.state_history.pop(0)  # pop first one
            self.state_history.append((c_t, h_t))
        elif self.model_type == "elman":
            h_t, s_t = self.model.soft_max_return_hidden_layer([word_window],
                                                               [pos_window])
            self.softmax_history.append(s_t)
            if len(self.state_history) == 20:
                self.state_history.pop(0)  # pop first one
            self.state_history.append(h_t)
        else:
            raise NotImplementedError("no softmax implemented for\
                                 {0} model".format(self.model_type))
        softmax = np.concatenate(self.softmax_history)

        # 3. do the decoding on the softmax
        if "disf" in self.args.tags:
            edit_tag = "<e/><cc/>" if "uttseg" in self.args.tags else "<e/>"
            # print self.tag_to_index_map[edit_tag]
            adjustsoftmax = np.concatenate((
                softmax,
                softmax[
                    :,
                    self.tag_to_index_map[edit_tag]].reshape(
                    softmax.shape[0],
                    1)), 1)
        else:
            adjustsoftmax = softmax
        last_n_timings = None if ((not self.args.use_timing_data) or
                                  not timing) \
            else get_last_n_features("timings", self.word_graph,
                                     len(self.word_graph) - 1,
                                     n=3)
        if not self.decoder:
            # no decoder, just get the arg max
            max_idx = np.argmax(adjustsoftmax[-1])
            # print max_idx
            max_tag = self.hmm_dict.keys()[
                self.hmm_dict.values().index(max_idx)]
            new_tags = [max_tag]
        else:
            new_tags = self.decoder.viterbi_incremental(
                adjustsoftmax, a_range=(len(adjustsoftmax) - 1,
                                        len(adjustsoftmax)),
                changed_suffix_only=True,
                timing_data=last_n_timings,
                words=[word])
        # print "new tags", new_tags
        prev_output_tags = deepcopy(self.output_tags)
        self.output_tags = self.output_tags[:len(self.output_tags) -
                                            (len(new_tags) -
                                             1)] + new_tags

        # 4. convert to standardized output format
        if "simple" in self.args.tags:
            for p in range(
                    len(self.output_tags) -
                    (len(new_tags) + 1),
                    len(self.output_tags)):
                rps = self.output_tags[p]
                self.output_tags[p] = rps.replace(
                    'rm-0',
                    'rps id="{}"'.format(p))
                if "<i" in self.output_tags[p]:
                    self.output_tags[p] = self.output_tags[p].\
                        replace("<e/>", "").replace("<i", "<e/><i")
        else:
            # new_words = [word]
            words = get_last_n_features("words", self.word_graph,
                                        len(self.word_graph) - 1,
                                        n=len(self.word_graph) -
                                        (self.window_size - 1))
            simple_conversion = False
            if simple_conversion:
                if "<e" in new_tags[-1]:
                    self.output_tags[-1] = "<e"
                elif "rm-" in new_tags[-1]:
                    self.output_tags[-1] = "<rps"
                else:
                    self.output_tags[-1] = "<f"
                if "rm-" in new_tags[-1]:
                    rps = re.findall("<rm-[0-9]+\/>", new_tags[-1], re.S)
                    for r in rps:  # should only be one
                        dist = int(r[r.find("-") + 1:-2])
                        for o in range(len(self.output_tags) - 2, -1, -1):
                            dist -= 1
                            if dist < 0:
                                break
                            if "<e" not in self.output_tags[o]:
                                self.output_tags[o] += "<rm"
            else:
                self.output_tags = \
                    convert_from_inc_disfluency_tags_to_eval_tags(
                        self.output_tags,
                        words,
                        start=len(self.output_tags) -
                        (len(new_tags)),
                        representation=self.args.tags)
        if diff_only:
            for i, old_new in enumerate(zip(prev_output_tags,
                                            self.output_tags)):
                old, new = old_new
                if old != new:
                    return self.output_tags[i:]
            return self.output_tags[len(prev_output_tags):]
        return self.output_tags

    def tag_utterance(self, utterance):
        """Tags entire utterance, only possible on models
        trained on unsegmented data.
        """
        if not self.args.utts_presegmented:
            raise NotImplementedError("Tagger trained on unsegmented data,\
            please call tag_prefix(words) instead.")
        # non segmenting
        self.reset()  # always starts in initial state
        if not self.args.pos:  # no pos tag model
            utterance = [(w, None, t) for w, p, t in utterance]
            # print "Warning: not using pos tags as not pos tag model"
        if not self.args.use_timing_data:
            utterance = [(w, p, None) for w, p, t in utterance]
            # print "Warning: not using timing durations as no timing model"
        for w, p, t in utterance:
            if self.args.pos:
                self.tag_new_word(w, pos=p, timing=t)
        return self.output_tags

    def rollback(self, backwards):
        super(DeepDisfluencyTagger, self).rollback(backwards)
        self.state_history = self.state_history[:len(self.state_history) -
                                                backwards]
        self.softmax_history = self.softmax_history[:
                                                    len(self.softmax_history) -
                                                    backwards]
        if self.decoder:
            self.decoder.rollback(backwards)

    def init_deep_model_internal_state(self):
        if self.model_type == "lstm":
            self.model.load_weights(c0=self.initial_c0_state,
                                    h0=self.initial_h0_state)
        elif self.model_type == "elman":
            self.model.load_weights(h0=self.initial_h0_state)

    def reset(self):
        super(DeepDisfluencyTagger, self).reset()
        self.word_graph = [("<s>", "<s>", 0)] * \
            (self.window_size - 1)
        self.state_history = []
        self.softmax_history = []
        if self.decoder:
            self.decoder.viterbi_init()
        self.init_deep_model_internal_state()

    def evaluate_fast_from_matrices(self, validation_matrices, tag_file,
                                    idx_to_label_dict):
        output = []
        true_y = []
        for v in validation_matrices:
            words_idx, pos_idx, extra, y, indices = v
            if extra:
                output.extend(self.model.classify_by_index(words_idx, indices,
                                                           pos_idx,
                                                           extra))
            else:
                output.extend(self.model.classify_by_index(words_idx, indices,
                                                           pos_idx))
            true_y.extend(y)
        p_r_f_tags = precision_recall_fscore_support(true_y,
                                                     output,
                                                     average='macro')
        tag_summary = classification_report(
            true_y, output,
            labels=[i for i in xrange(len(idx_to_label_dict.items()))],
            target_names=[
                idx_to_label_dict[i]
                for i in xrange(len(idx_to_label_dict.items()))
            ]
        )
        print tag_summary
        results = {"f1_rmtto": p_r_f_tags[2], "f1_rm": p_r_f_tags[2],
                   "f1_tto1": p_r_f_tags[2], "f1_tto2": p_r_f_tags[2]}

        results.update({
            'f1_tags': p_r_f_tags[2],
            'tag_summary': tag_summary
        })
        return results

    def train_net(self, train_dialogues_filepath=None,
                  validation_dialogues_filepath=None,
                  model_dir=None,
                  tag_accuracy_file_path=None):
        """Train the internal deep learning model
        from a list of dialogue matrices.
        """
        tag_accuracy_file = open(tag_accuracy_file_path, "a")
        print "Verifying files..."
        for filepath in [train_dialogues_filepath,
                         validation_dialogues_filepath]:
            if not verify_dialogue_data_matrices_from_folder(
                    filepath,
                    word_dict=self.word_to_index_map,
                    pos_dict=self.pos_to_index_map,
                    tag_dict=self.tag_to_index_map,
                    n_lm=self.args.n_language_model_features,
                    n_acoustic=self.args.n_acoustic_features):
                raise Exception("Dialogue vectors in wrong format!\
                See README.md.")
        lr = self.args.lr  # even if decay, start with specific lr
        n_extra = self.args.n_language_model_features + \
            self.args.n_acoustic_features
        # validation matrices filepath much smaller so can store these
        # and preprocess them all:
        validation_matrices = [np.load(
            validation_dialogues_filepath + "/" + fp)
            for fp in os.listdir(
            validation_dialogues_filepath)]
        validation_matrices = [dialogue_data_and_indices_from_matrix(
            d_matrix,
            n_extra,
            pre_seg=self.args.utts_presegmented,
            window_size=self.window_size,
            bs=self.args.bs,
            tag_rep=self.args.tags,
            tag_to_idx_map=self.tag_to_index_map,
            in_utterances=self.args.utts_presegmented)
            for d_matrix in validation_matrices
        ]
        idx_2_label_dict = {v: k for k, v in self.tag_to_index_map.items()}
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        start = 1  # by default start from the first epoch
        best_score = 0
        best_epoch = 0
        print "Net training started..."
        for e in range(start, self.args.n_epochs + 1):
            tic = time.time()
            epoch_folder = model_dir + "/epoch_{}".format(e)
            if not os.path.exists(epoch_folder):
                os.mkdir(epoch_folder)
            train_loss = 0
            # TODO IO is slow, where the memory allows do in one
            load_separately = True
            test = False
            if load_separately:
                for i, dialogue_f in enumerate(
                        os.listdir(train_dialogues_filepath)):
                    if test and i > 3:
                        break
                    print dialogue_f
                    d_matrix = np.load(train_dialogues_filepath + "/" +
                                       dialogue_f)
                    word_idx, pos_idx, extra, y, indices = \
                        dialogue_data_and_indices_from_matrix(
                            d_matrix,
                            n_extra,
                            window_size=self.window_size,
                            bs=self.args.bs,
                            pre_seg=self.args.utts_presegmented
                        )
                    # for i in range(len(indices)):
                    #     print i, word_idx[i], pos_idx[i], \
                    #     y[i], indices[i]
                    train_loss += self.model.fit(word_idx,
                                                 y,
                                                 lr,
                                                 indices,
                                                 pos_idx=pos_idx,
                                                 extra_features=extra)
                    print '[learning] file %i >>' % (i + 1),\
                        'completed in %.2f (sec) <<\r' % (time.time() - tic)
            # save the initial states we've learned to override the random
            self.initial_h0_state = self.model.h0.get_value()
            if self.args.model_type == "lstm":
                self.initial_c0_state = self.model.c0.get_value()
            # reset and evaluate simply
            self.reset()
            results = self.evaluate_fast_from_matrices(
                validation_matrices,
                tag_accuracy_file,
                idx_to_label_dict=idx_2_label_dict
            )
            val_score = results['f1_tags']  # TODO get best score type
            print "epoch training loss", train_loss
            print '[learning] epoch %i >>' % (e),\
                'completed in %.2f (sec) <<\r' % (time.time() - tic)
            print "validation score", val_score
            tag_accuracy_file.write(str(e) + "\n" + results['tag_summary'] +
                                    "\n%%%%%%%%%%\n")
            tag_accuracy_file.flush()
            print "saving model..."
            self.model.save(epoch_folder)  # Epoch file dump
            # checking patience and decay, if applicable
            # stopping criterion
            if val_score > best_score:
                self.model.save(model_dir)
                best_score = val_score
                print 'NEW BEST raw labels at epoch ', e, 'best valid',\
                    best_score
                best_epoch = e
            # stopping criteria = if no improvement in 10 epochs
            if e - best_epoch >= 10:
                print "stopping, no improvement in 10 epochs"
                break
            if self.args.decay and (e - best_epoch) > 1:
                # just a steady decay if things aren't improving for 2 epochs
                # a hidden hyperparameter
                decay_rate = 0.85
                lr *= decay_rate
                print "learning rate decayed, now ", lr
            if lr < 1e-5:
                print "stopping, below learning rate threshold"
                break
            print '[learning and testing] epoch %i >>' % (e),\
                'completed in %.2f (sec) <<\r' % (time.time() - tic)

        print 'BEST RESULT: epoch', best_epoch, 'valid score', best_score
        tag_accuracy_file.close()
        return best_epoch

    def get_output_tags(self, with_words=False):
        if with_words:
            return zip(self.output_tags,
                       self.word_graph[self.window_size - 1:])
        return self.output_tags

    def incremental_output_from_file(self, source_file_path,
                                     target_file_path=None,
                                     is_asr_results_file=False):
        """Return the incremental output in an increco style
        given the incoming words + POS. E.g.:

        Speaker: KB3_1

        Time: 1.50
        KB3_1:1    0.00    1.12    $unc$yes    NNP    <f/><tc/>

        Time: 2.10
        KB3_1:1    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
        KB3_1:2    1.12    2.00     because    IN    <rps id="1"/><cc/>

        Time: 2.5
        KB3_1:2    1.12    2.00     because    IN    <rps id="1"/><rpndel id="1"/><cc/>

        from an ASR increco style input without the POStags:

        or a normal style disfluency dectection ground truth corpus:

        Speaker: KB3_1
        KB3_1:1    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
        KB3_1:2    1.12    2.00     $because    IN    <rps id="1"/><cc/>
        KB3_1:3    2.00    3.00    because    IN    <f/><cc/>
        KB3_1:4    3.00    4.00    theres    EXVBZ    <f/><cc/>
        KB3_1:6    4.00    5.00    a    DT    <f/><cc/>
        KB3_1:7    6.00    7.10    pause    NN    <f/><cc/>


        :param source_file_path: str, file path to the input file
        :param target_file_path: str, file path to output in the above format
        :param is_asr_results_file: bool, whether the input is increco style
        """
        if target_file_path:
            target_file = open(target_file_path, "w")
        if not self.args.do_utt_segmentation:
            print "not doing utt seg, using pre-segmented file"
        if is_asr_results_file:
            return NotImplementedError
        if 'timings' in source_file_path:
            print "input file has timings"
            if not is_asr_results_file:
                dialogues = []
                IDs, timings, words, pos_tags, labels = \
                    get_tag_data_from_corpus_file(source_file_path)
                for dialogue, a, b, c, d in zip(IDs,
                                                timings,
                                                words,
                                                pos_tags,
                                                labels):
                    dialogues.append((dialogue, (a, b, c, d)))
        else:
            print "no timings in input file, creating fake timings"
            raise NotImplementedError

        for speaker, speaker_data in dialogues:
            # if "4565" in speaker: quit()
            print speaker
            self.reset()  # reset at the beginning of each dialogue
            if target_file_path:
                target_file.write("Speaker: " + str(speaker) + "\n\n")
            timing_data, lex_data, pos_data, labels = speaker_data
            # iterate through the utterances
            # utt_idx = -1
            current_time = 0
            for i in range(0, len(timing_data)):
                # print i, timing_data[i]
                _, end = timing_data[i]
                if (not self.args.do_utt_segmentation) \
                    and self.args.utts_presegmented \
                        and "<t" in labels[i]:
                    self.reset()  # reset after each utt if non pre-seg
                # utt_idx = frames[i]
                timing = None
                if 'timings' in source_file_path and self.args.use_timing_data:
                    timing = end - current_time
                word = lex_data[i]
                pos = pos_data[i]
                diff = self.tag_new_word(word, pos, timing,
                                         diff_only=True,
                                         rollback=0)
                current_time = end
                if target_file_path:
                    target_file.write("Time: " + str(current_time) + "\n")
                    new_words = lex_data[i - (len(diff) - 1):i + 1]
                    new_pos = pos_data[i - (len(diff) - 1):i + 1]
                    new_timings = timing_data[i - (len(diff) - 1):i + 1]
                    for t, w, p, tag in zip(new_timings,
                                            new_words,
                                            new_pos,
                                            diff):
                        target_file.write("\t".join([str(t[0]),
                                                     str(t[1]),
                                                     w,
                                                     p,
                                                     tag]))
                        target_file.write("\n")
                    target_file.write("\n")
            target_file.write("\n")

    def train_decoder(self, tag_file):
        raise NotImplementedError

    def save_decoder_model(self, dir_path):
        raise NotImplementedError
