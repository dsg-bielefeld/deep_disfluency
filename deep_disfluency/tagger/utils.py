import argparse

config_header = [
    'exp_id',  # experiment id
    'model_type',  # can be elman/lstm/mt_elman/mt_lstm
    'lr',  # learning rate
    'decay',  # decay on the learning rate if improvement stops
    'seed',  # random seed
    'window',  # number of words in the context window (backwards only)
    'bs',  # number of backprop through time steps
    'emb_dimension',  # dimension of word embedding
    'n_hidden',  # number of hidden units
    'n_epochs',  # maximum number of epochs
    'train_data',  # which training data
    'partial_words',  # whether partial words are in the data
    'loss_function',  # default will be nll, unlikely to change
    'reg',  # regularization type
    'pos',  # whether pos tags or not
    'n_acoustic_features',  # number of acoustic features used per word
    'n_language_model_features',  # number of language model features used
    'embeddings',  # embedding files, if any
    'update_embeddings',  # whether the embeddings should be updates
    'batch_size',  # batch size, 'word' or 'utterance'
    'word_rep',  # the word to index mapping filename
    'pos_rep',  # the pos tag to index mapping filename
    'tags',  # the output tag representations used
    'decoder_type',  # which type of decoder
    'utts_presegmented',  # whether utterances are pre-segmented
    'do_utt_segmentation'  # whether we do combined end of utt detection
]


class SimpleArgs(object):
    pass


def process_arguments(config=None,
                      exp_id=None,
                      heldout_file="../data/disfluency_detection/" +
                      "switchboard/swbd_heldout_partial_data.csv",
                      test_file="../data/disfluency_detection/" +
                      "switchboard/swbd_test_partial_data.csv",
                      use_saved=None,
                      hmm=None,
                      verbose=True):
    """Loads arguments for an experiment from a config file

    Keyword arguments:
    config -- the config file location, default None
    exp_id -- the experiment ID name, default None
    """

# # the legacy argparse version with explanations:
#     parser = argparse.ArgumentParser(description='This script trains a RNN\
#         for disfluency detection and saves the best models and results to
#          disk.')
#     parser.add_argument('-c', '--config', type=str,
#                         help='The location of the config file.',
#                         default=config)
#     parser.add_argument('-e', '--exp_id', type=str,
#                         help='The experiment number from which to load \
#                             arguments from the config file.',
#                         default=exp_id)
#     parser.add_argument('-v', '--heldout_file', type=str,
#                         help='The path to the validation file.',
#                         default=heldout_file)
#     parser.add_argument('-t', '--test_file', type=str,
#                         help='The path to the test file.',
#                         default=test_file)
#     parser.add_argument('-m', '--use_saved_model', type=int,
#                       help='Epoch number of the pre-trained model to load.',
#                         default=use_saved)
#     parser.add_argument('-hmm', '--decoder_file', type=str,
#                         help='Path to the hmm file.',
#                         default=hmm)
#     parser.add_argument('-verb', '--verbose', type=bool,
#                         help='Whether to output training progress.',
#                         default=verbose)
#     args = parser.parse_args()
    # newer simple version:
    args = SimpleArgs()
    setattr(args, "config", config)
    setattr(args, "exp_id", exp_id)
    setattr(args, "heldout_file", test_file)
    setattr(args, "test_file", test_file)
    setattr(args, "use_saved_model", use_saved)
    setattr(args, "decoder_file", hmm)
    setattr(args, "verbose", verbose)

    if args.config:
        for line in open(args.config):
            # print line
            features = line.strip("\n").split(",")
            if features[0] != str(args.exp_id):
                continue
            for i in range(1, len(config_header)):
                feat_value = features[i].strip()  # if string
                if feat_value == 'None':
                    feat_value = None
                elif feat_value == 'True':
                    feat_value = True
                elif feat_value == 'False':
                    feat_value = False
                elif config_header[i] in ['lr']:
                    feat_value = float(feat_value)
                elif config_header[i] in ['seed', 'window', 'bs',
                                          'emb_dimension', 'n_hidden',
                                          'n_epochs',
                                          'n_acoustic_features',
                                          'n_language_model_features'
                                          ]:
                    feat_value = int(feat_value)
                # print config_header[i], feat_value
                setattr(args, config_header[i], feat_value)
    return args


def get_last_n_features(feature, current_words, idx, n=3):
    """For the purposes of timing info, get the timing, word or pos
    values  of the last n words (default = 3).
    """
    if feature == "words":
        position = 0
    elif feature == "POS":
        position = 1
    elif feature == "timings":
        position = 2
    else:
        raise Exception("Unknown feature {}".format(feature))
    start = max(0, (idx - n) + 1)
    # print start, idx + 1
    return [triple[position] for triple in
            current_words[start: idx + 1]]


def simulate_increco_data(frame, acoustic_data, lexical_data, pos_data):
    """For transcripts + timings, create tuples of single hypotheses
    to simulate perfect ASR at the end of each word.
    """
    new_lexical_data = []
    new_pos_data = []
    new_acoustic_data = []
    current_time = 0
    for my_frame, acoust, word, pos in zip(frame, acoustic_data,
                                           lexical_data, pos_data):
        new_lexical_data.append([(word, current_time / 100, my_frame / 100)])
        current_time = my_frame
        new_pos_data.append([pos])
        new_acoustic_data.append([acoust])
    return new_acoustic_data, new_lexical_data, new_pos_data
