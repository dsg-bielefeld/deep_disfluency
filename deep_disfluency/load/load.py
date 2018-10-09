import urllib
import logging
import os
import csv
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

from deep_disfluency.utils.tools import convert_from_eval_tags_to_inc_disfluency_tags
from deep_disfluency.utils.tools import add_word_continuation_tags
from deep_disfluency.utils.tools import context_win_backwards
from deep_disfluency.utils.tools import indices_from_length
from deep_disfluency.utils.tools import convert_to_simple_label
from deep_disfluency.utils.tools import convert_from_full_tag_set_to_idx
from deep_disfluency.feature_extraction.feature_utils \
    import fill_in_time_approximations

logger = logging.getLogger(__name__)

DATAPREFIX = os.path.dirname(os.path.realpath(__file__)) + os.path.sep +\
    '..' + os.path.sep + 'data'


def open_with_pandas_read_csv(filename, header=None, delim="\t"):
    df = pd.read_csv(filename, sep=delim, header=header)
    data = df.values
    return data


def load_data_from_array(data, n_acoust, cs=2, bs=9,
                         tags="disf1_uttseg_simple",
                         full_idx_to_label_dict=None, idx_to_label=None):
    """Returns a tuple of (frames, acoustic_data, lex_data, pos_data,
    indices, labels )
    From a numpy array with all those bits of info in them.
    """

    frames = data[0, :]
    acoustic_data = data[1: n_acoust + 1, :]  # 1 to n_acoust+1 is all acoustic
    # print 'acoustic_data', acoustic_data.shape, acoustic_data[0][0:10]
    lex_data = data[-3, :]  # NB these should be windows from the outset really?
    # print 'lex data', lex_data.shape, lex_data[0:10]
    pos_data = data[-2, :]  # NB these should be windows from the outset really?
    # print 'pos data', pos_data.shape, pos_data[0:10]
    labels = data[-1, :]
    # print 'labels', labels.shape, labels[0:10]

    # print 'ADJUSTED'
    lex_data = np.asarray(context_win_backwards(lex_data, cs)).astype('int32')
    # print 'lex_data', lex_data.shape, lex_data[0:10]
    pos_data = np.asarray(context_win_backwards(pos_data, cs)).astype('int32')
    # print 'pos_data', pos_data.shape, pos_data[0:10]
    acoustic_data = np.swapaxes(acoustic_data, 0, 1)
    # print 'acoustic_data', acoustic_data.shape

    # raw_input()

    # these need to be created
    # indices = data[:, -3:-1] #anti-penult = start, penult = stop; columns show how the slicing of context should be done in training
    labels = list(data[-1, :])  # last one is the label column
    if "simple" not in tags:  # TODO this is a hack as the data should be in the right format remove after exps.
        for i in range(0, len(labels)):
            # if 'simple' in tags:
            labels[i] = convert_from_full_tag_set_to_idx(
                full_idx_to_label_dict[labels[i]], tags, idx_to_label)
            # now map the labels to the correct indices

    labels = np.asarray(labels)
    # print 'labels', labels.shape, labels[0:10]

    # print 'CREATED'
    indices = np.asarray(indices_from_length(len(lex_data), bs, 0)).astype('int32')
    # print 'indices', indices.shape, indices[0:10]

    # raw_input()
    return frames, acoustic_data, lex_data, pos_data, indices, labels


def switchboard_data(train_data=None, tags=None):
    """Returns the training, validation and test data and the tag and word index representations for switchboard"""

    ftrain = open(os.path.sep.join([DATAPREFIX, 'disfluency_detection', 'switchboard', train_data + '_partial_data.csv']))
    fvalid = open(os.path.sep.join([DATAPREFIX, 'disfluency_detection', 'switchboard', 'swbd_heldout_partial_data.csv']))
    ftest = open(os.path.sep.join([DATAPREFIX, 'disfluency_detection', 'switchboard', 'swbd_test_partial_data.csv']))
    test_dict = defaultdict()
    test_dict['words2idx'] = load_word_rep(os.path.sep.join([DATAPREFIX, 'tag_representations', 'swbd_word_rep.csv']))
    test_dict['pos2idx'] = load_word_rep(os.path.sep.join([DATAPREFIX, 'tag_representations', 'swbd_pos_rep.csv']))
    test_dict['labels2idx'] = load_tags(os.path.sep.join([DATAPREFIX, 'tag_representations', 'swbd' + tags + '_tags.csv']))

    # load testing corpora
    train = load_data_from_file(ftrain, test_dict['words2idx'], test_dict['pos2idx'], test_dict['labels2idx'], representation=tags)
    valid = load_data_from_file(fvalid, test_dict['words2idx'], test_dict['pos2idx'], test_dict['labels2idx'], representation=tags)
    test = load_data_from_file(ftest, test_dict['words2idx'], test_dict['pos2idx'], test_dict['labels2idx'], representation=tags)

    return train, valid, test, test_dict


def load_word_rep(filepath, dimension=None, word_rep_type="one_hot"):
    """Returns a word_rep_dictionary from word(string) indicating an index by an integer"""
    word_rep_dictionary = None
    if word_rep_type == "one_hot":
        word_rep_dictionary = defaultdict(int)  # TODO could use sparse matrices instead?
        f = open(filepath)
        for line in f:
            l = line.strip("\n").split(",")
            word_rep_dictionary[l[1]] = int(l[0])
        f.close()
    elif word_rep_type == "word_freq_count":
        raise NotImplementedError()
    elif word_rep_type == "neural_word":
        raise NotImplementedError()
    return word_rep_dictionary


def load_tags(filepath):
    """Returns a tag dictionary from word to a n int indicating index
    by an integer
    """
    tag_dictionary = defaultdict(int)
    f = open(filepath)
    for line in f:
        l = line.strip('\n').split(",")
        tag_dictionary[l[1]] = int(l[0])
    f.close()
    return tag_dictionary


def load_data_from_file(f, word_rep, pos_rep, tag_rep, representation="1", limit=8, n_seq=None):
    """Loads from file into five lists of arrays of equal length:
    one for utterance iDs (IDs))
    one for the timings of the tags (start, stop)
    one for words (seq),
    one for pos (pos_seq)
    one for tags (targets).

    Converts them into arrays of one-hot representations."""

    print "loading data", f.name
    count_seq = 0
    #count_step = 0
    IDs = []
    seq = []
    pos_seq = []
    targets = []
    timings = []

    reader = csv.reader(f, delimiter='\t')
    counter = 0
    utt_reference = ""
    currentWords = []
    currentPOS = []
    currentTags = []
    currentTimings = []
    current_fake_time = 0  # marks the current fake time for the dialogue (i.e. end of word)
    current_dialogue = ""

    # corpus = "" # can write to file
    for ref, timing, word, postag, disftag in reader:  # mixture of POS and Words
        # TODO, for now 'fake' timing will increment by one each time
        counter += 1

        if not ref == "":
            if count_seq > 0:  # do not reset the first time
                # convert to the inc tags
                if "0" in representation:  # turn taking only
                    currentTags = [""] * len(currentTags)
                else:
                    currentTags = convert_from_eval_tags_to_inc_disfluency_tags(currentTags, currentWords, representation=representation, limit=limit)
                if 'trp' in representation:
                    currentTags = add_word_continuation_tags(currentTags)
                if 'simple' in representation:
                    currentTags = map(lambda x: convert_to_simple_label(x, rep=representation), currentTags)
                # corpus+=utt_reference #write data to a file for checking
                # convert to vectors
                words = []
                pos_tags = []
                tags = []
                for i in range(0, len(currentTags)):
                    w = word_rep.get(currentWords[i])
                    pos = pos_rep.get(currentPOS[i])
                    tag = tag_rep.get(currentTags[i])  # NB POS tags in switchboard at l[2]
                    if w == None:
                        logging.info("No word rep for :" + currentWords[i])
                        w = word_rep.get("<unk>")
                    if pos == None:
                        logging.info("No pos rep for :" + currentPOS[i])
                        pos = pos_rep.get("<unk>")
                    if tag == None:
                        logging.info("No tag rep for:" + currentTags[i])
                        print utt_reference, currentTags, words
                        raise Exception("No tag rep for:" + currentTags[i])
                    words.append(w)
                    pos_tags.append(pos)
                    tags.append(tag)
                x = np.asarray(words)
                p = np.asarray(pos_tags)
                y = np.asarray(tags)
                seq.append(x)
                pos_seq.append(p)
                targets.append(y)
                IDs.append(utt_reference)
                timings.append(tuple(currentTimings))
                # reset the words
                currentWords = []
                currentPOS = []
                currentTags = []
                currentTimings = []
            # set the utterance reference
            count_seq += 1
            utt_reference = ref
            if not utt_reference.split(":")[0] == current_dialogue:
                current_dialogue = utt_reference.split(":")[0]
                current_fake_time = 0  # TODO fake for now- reset the current beginning of word time
        currentWords.append(word)
        currentPOS.append(postag)
        currentTags.append(disftag)
        currentTimings.append((current_fake_time, current_fake_time + 1))
        current_fake_time += 1
    # flush
    if not currentWords == []:
        if "0" in representation:  # turn taking only
            currentTags = [""] * len(currentTags)
        else:
            currentTags = convert_from_eval_tags_to_inc_disfluency_tags(currentTags, currentWords, representation=representation, limit=limit)
        if 'trp' in representation:
            currentTags = add_word_continuation_tags(currentTags)
        if 'simple' in representation:
            currentTags = map(lambda x: convert_to_simple_label(x, rep=representation), currentTags)
        words = []
        pos_tags = []
        tags = []
        for i in range(0, len(currentTags)):
            w = word_rep.get(currentWords[i])
            pos = pos_rep.get(currentPOS[i])
            tag = tag_rep.get(currentTags[i])  # NB POS tags in switchboard at l[2]
            if w == None:
                logging.info("No word rep for :" + currentWords[i])
                w = word_rep.get("<unk>")
            if pos == None:
                logging.info("No pos rep for :" + currentPOS[i])
                pos = pos_rep.get("<unk>")
            if tag == None:
                logging.info("No tag rep for:" + currentTags[i])
                print utt_reference, currentTags, words
                raise Exception("No tag rep for:" + currentTags[i])
            words.append(w)
            pos_tags.append(pos)
            tags.append(tag)
        x = np.asarray(words)
        p = np.asarray(pos_tags)
        y = np.asarray(tags)
        seq.append(x)
        pos_seq.append(p)
        targets.append(y)
        IDs.append(utt_reference)
        timings.append(tuple(currentTimings))

    assert len(seq) == len(targets) == len(pos_seq)
    print "loaded " + str(len(seq)) + " sequences"
    f.close()
    return (IDs, timings, seq, pos_seq, targets)


def load_increco_data_from_file(increco_filename, word_2_ind, pos_2_ind):
    """Loads increco style data from file.
    For now returns word and pos data only"""
    all_speakers = []
    lex_data = []
    pos_data = []
    frames = []
    latest_increco = []
    latest_pos = []
    file = open(increco_filename)
    started = False
    conv_no = ""
    prev_word = -1
    prev_pos = -1
    for line in file:
        if "Time:" in line:
            if not latest_increco == []:
                lex_data.append(deepcopy(latest_increco))
                pos_data.append(deepcopy(latest_pos))
            latest_increco = []
            latest_pos = []
            continue
        if "File:" in line:
            if not started:
                started = True
            else:
                # flush
                if not latest_increco == []:
                    lex_data.append(deepcopy(latest_increco))
                    pos_data.append(deepcopy(latest_pos))
                # fake
                # print lex_data
                frames = [x[-1][-1] for x in lex_data]  # last word end time
                acoustic_data = [0, ] * len(lex_data)  # fakes..
                indices = [0, ] * len(lex_data)
                labels = [0, ] * len(lex_data)
                all_speakers.append((conv_no, (frames, acoustic_data, lex_data, pos_data, indices, labels)))
                # reset
                lex_data = []
                pos_data = []
                latest_increco = []
                latest_pos = []
                prev_word = -1
                prev_pos = -1

            conv_no = line.strip("\n").replace("File: ", "")
            continue
        if line.strip("\n") == "":
            continue
        spl = line.strip("\n").split("\t")
        start = float(spl[0])
        end = float(spl[1])
        if spl[2] in word_2_ind.keys():
            word = word_2_ind[spl[2]]
        else:
            word = word_2_ind["<unk>"]
        if spl[3] in pos_2_ind.keys():
            pos = pos_2_ind[spl[3]]
        else:
            pos = pos_2_ind["<unk>"]
        latest_increco.append(([prev_word, word], start, end))
        latest_pos.append(deepcopy([prev_pos, pos]))
        prev_word = word
        prev_pos = pos

    # flush
    if not latest_increco == []:
        lex_data.append(latest_increco)
        pos_data.append(latest_pos)
    frames = [x[-1][-1] for x in lex_data]  # last word end time
    acoustic_data = [0, ] * len(lex_data)  # fakes..
    indices = [0, ] * len(lex_data)
    labels = [0, ] * len(lex_data)
    all_speakers.append((conv_no, (frames, acoustic_data, lex_data, pos_data, indices, labels)))
    print len(all_speakers), "speakers with increco input"
    return all_speakers


def load_data_from_timings_file(filename, word_2_ind, pos_2_ind):
    """Loads from disfluency detection with timings file."""
    all_speakers = []
    lex_data = []
    pos_data = []
    frames = []
    labels = []

    latest_increco = []
    latest_pos = []
    latest_labels = []

    file = open(filename)
    started = False
    conv_no = ""
    prev_word = -1
    prev_pos = -1
    for line in file:
        if "File:" in line:
            if not started:
                started = True
            else:
                # flush
                # print line
                if not latest_increco == []:
                    shift = -1
                    for i in range(0, len(latest_increco)):
                        triple = latest_increco[i]
                        if triple[1] == triple[2]:
                            # print "same timing!", triple
                            shift = i
                            break
                    if shift > -1:
                        latest_increco = fill_in_time_approximations(latest_increco, shift)
                    lex_data.append(deepcopy(latest_increco))
                    pos_data.append(deepcopy(latest_pos))
                    # convert to the disfluency tags for this
                    #latest_labels = convertFromEvalTagsToIncDisfluencyTags()
                    labels.extend(deepcopy(latest_labels))
                # fake
                # print lex_data
                frames = [x[-1][-1] for x in lex_data]  # last word end time
                acoustic_data = [0, ] * len(lex_data)  # fakes..
                indices = [0, ] * len(lex_data)

                all_speakers.append((conv_no, (frames, acoustic_data, lex_data, pos_data, indices, labels)))
                # reset
                lex_data = []
                pos_data = []
                latest_increco = []
                latest_pos = []
                latest_labels = []
                prev_word = -1
                prev_pos = -1

            conv_no = line.strip("\n").replace("File: ", "")
            continue
        if line.strip("\n") == "":
            continue
        spl = line.strip("\n").split("\t")
        start = float(spl[1])
        end = float(spl[2])
        if spl[3] in word_2_ind.keys():
            word = word_2_ind[spl[3]]
        else:
            word = word_2_ind["<unk>"]
        if spl[4] in pos_2_ind.keys():
            pos = pos_2_ind[spl[4]]
        else:
            pos = pos_2_ind["<unk>"]
        # need to convert to the right rep here
        tag = spl[5]

        latest_increco.append(([prev_word, word], start, end))
        latest_pos.append(deepcopy([prev_pos, pos]))
        latest_labels.append(tag)
        prev_word = word
        prev_pos = pos

    # flush
    if not latest_increco == []:
        shift = -1
        for i in range(0, len(latest_increco)):
            triple = latest_increco[i]
            if triple[1] == triple[2]:
                shift = i
                break
        if shift > -1:
            latest_increco = fill_in_time_approximations(latest_increco, shift)
        lex_data.append(latest_increco)
        pos_data.append(latest_pos)
        labels.extend(latest_labels)
    frames = [x[-1][-1] for x in lex_data]  # last word end time
    acoustic_data = [0, ] * len(lex_data)  # fakes..
    indices = [0, ] * len(lex_data)
    all_speakers.append((conv_no, (frames, acoustic_data, lex_data, pos_data, indices, labels)))
    print len(all_speakers), "speakers with timings input"
    return all_speakers


def get_tag_data_from_corpus_file(f, representation="1", limit=8):
    """Loads from file into five lists of lists of strings of equal length:
    one for utterance iDs (IDs))
    one for word timings of the targets (start,stop)
    one for words (seq),
    one for pos (pos_seq)
    one for tags (targets).

    NB this does not convert them into one-hot arrays, just outputs lists of string tags in GOLD form."""

    f = open(f)
    print "loading data", f.name
    count_seq = 0
    IDs = []
    seq = []
    pos_seq = []
    targets = []
    timings = []
    currentTimings = []
    current_fake_time = 0  # marks the current fake time for the dialogue (i.e. end of word)
    current_dialogue = ""

    reader = csv.reader(f, delimiter='\t')
    counter = 0
    utt_reference = ""
    currentWords = []
    currentPOS = []
    currentTags = []
    current_fake_time = 0

    # corpus = "" # can write to file
    for ref, timing, word, postag, disftag in reader:  # mixture of POS and Words
        counter += 1
        if not ref == "":
            if count_seq > 0:  # do not reset the first time
                # convert to the inc tags
                #currentTags = convertFromEvalTagsToIncDisfluencyTags(currentTags, currentWords, representation, limit)
                if 'trp' in representation:
                    currentTags = add_word_continuation_tags(currentTags)
                if 'simple' in representation:
                    currentTags = map(lambda x: convert_to_simple_label(x), currentTags)
                # corpus+=utt_reference #write data to a file for checking
                # convert to vectors
                seq.append(tuple(currentWords))
                pos_seq.append(tuple(currentPOS))
                targets.append(tuple(currentTags))
                IDs.append(utt_reference)
                timings.append(tuple(currentTimings))
                # reset the words
                currentWords = []
                currentPOS = []
                currentTags = []
                currentTimings = []
            # set the utterance reference
            count_seq += 1
            utt_reference = ref
            if not utt_reference.split(":")[0] == current_dialogue:
                current_dialogue = utt_reference.split(":")[0]
                current_fake_time = 0  # TODO fake for now- reset the current beginning of word time
        currentWords.append(word)
        currentPOS.append(postag)
        currentTags.append(disftag)
        currentTimings.append((current_fake_time, current_fake_time + 1))
        current_fake_time += 1
    # flush
    if not currentWords == []:
        #currentTags = convertFromEvalTagsToIncDisfluencyTags(currentTags, currentWords, limit=8)
        if 'trp' in representation:
            currentTags = add_word_continuation_tags(currentTags)
        if 'simple' in representation:
            currentTags = map(lambda x: convert_to_simple_label(x), currentTags)
        seq.append(tuple(currentWords))
        pos_seq.append(tuple(currentPOS))
        targets.append(tuple(currentTags))
        IDs.append(utt_reference)
        timings.append(currentTimings)

    assert len(seq) == len(targets) == len(pos_seq)
    print "loaded " + str(len(seq)) + " sequences"
    f.close()
    return (IDs, timings, seq, pos_seq, targets)


def download(origin):
    '''download the corresponding file from origin
    '''
    print 'Downloading data from %s' % origin
    name = origin.split('/')[-1]
    urllib.urlretrieve(origin, name)


if __name__ == '__main__':
    print load_word_rep("../data/tag_representations/swbd_pos_rep.csv")
