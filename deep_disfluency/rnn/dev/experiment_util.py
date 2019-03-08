
import argparse
import numpy as np
from math import log
import pickle
import os
from copy import deepcopy
from collections import defaultdict
import theano.tensor as T
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from nltk.tag import CRFTagger

import sys
sys.path.append('../') #path to the src files

from language_model.ngram_language_model import KneserNeySmoothingModel
from utils.tools import convertFromIncDisfluencyTagsToEvalTags
from load.load import load_tags
from rnn.elman import Elman
from rnn.lstm import LSTM
from decoder.hmm import FirstOrderHMM


def get_diff_and_new_prefix(current, newprefix, verbose=False):
    """Only get the different right frontier according to the timings
    and change the current hypotheses.
    """
    if verbose:
        print("current", current)
        print("newprefix", newprefix)
    rollback = 0
    original_length = len(current)
    original_current = deepcopy(current)
    for i in range(len(current)-1, -2, -1):
        if verbose:
            print("oooo", newprefix[0])
            if not current == []:
                print(current[i])
        if i == -1 or (float(newprefix[0][1]) >= float(current[i][2])):
            if i == len(current)-1:
                current = current + newprefix
                break
            k = 0
            marker = i+1
            for j in range(i+1, len(current)):
                if k == len(newprefix):
                    break
                if verbose:
                    print("...", j, k, current[j], newprefix[k], len(newprefix))
                if not current[j] == newprefix[k]:
                    break
                else:
                    if verbose:
                        print("repeat")
                    k += 1
                    marker = j+1
            rollback = original_length - marker
            current = current[:marker] + newprefix[k:]
            newprefix = newprefix[k:]
            break
    if newprefix == []:
        rollback = 0  # just no rollback if no prefix
        current = original_current  # reset the current
    if verbose:
        print("current after call", current)
        print("newprefix after call", newprefix)
        print("rollback after call", rollback)
    return (current, newprefix, rollback)


def div(enum, denom):
    if denom == 0.0 or enum == 0.0:
        return 0.0
    return enum/denom


def get_last_n_features(feature, current_words, idx, n=3):
    """For the purposes of timing info, get the timing, word or pos
    values  of the last n words (default = 3).
    """
    if feature == "timing":
        last = 0
        timings = []
        if idx > n:
            last = current_words[idx-n][2]
        for _, _, e in current_words[max(0, (idx - n) + 1): idx + 1]:
            timings.append((e-last)*100)
            last = e
        while len(timings) < n:
            timings = [0] + timings
        return timings
    else:
        # words or pos
        start = max(0, (idx - n) + 1)
        print(start, idx + 1)
        position = 1 if feature == "POS" else 0
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
        new_lexical_data.append([(word, current_time/100, my_frame/100)])
        current_time = my_frame
        new_pos_data.append([pos])
        new_acoustic_data.append([acoust])
    return new_acoustic_data, new_lexical_data, new_pos_data


def save_predictions_and_quick_eval(predictions_filename=None,
                                    groundtruth_filename=None,
                                    model=None,
                                    hmm=None,
                                    dialogues=None,
                                    idx_to_label_dict=None,
                                    idx_to_word_dict=None,
                                    incremental_eval=False,
                                    s=None,
                                    increco_style=False):
    """Method to output a csv file with an rnns softmax predictions over
    each class for step in each file in dialogues.
    Does a rough evaluation for the purposes of the stopping criterion.
    i.e. No revision of the distributions due to HMM.
    """
    label_to_idx_dict = {v: k for k, v in list(idx_to_label_dict.items())}
    classes = [idx_to_label_dict[idx]
               for idx in sorted(idx_to_label_dict.keys())]
    raw_predictions_file = open(predictions_filename, "w")
    raw_predictions_header = "file_name,time_step,"+",".\
        join(classes)+",arg_max,y"
    raw_predictions_file.write(raw_predictions_header+"\n")
    if incremental_eval:
        inc_predictions_file = open(predictions_filename.replace(
            ".csv", ".increco"), "w")
    raw_tag_labels = []  # just the numbers referring to the classes (labels)
    raw_tag_predictions = []  # just the numbers referring to (predictions)
    predictions = []  # just the local predictions for the current dialogue
    prev_predictions = []  # saves the last step's predictions
    labels = []
    class_loss = defaultdict(list)
    loss = 0.0
    total = 0
    count = 0  # for testing
    for d in dialogues:
        count += 1
        predictions = []
        prev_predictions = []
        labels = []
        dialogue_id, dialogue_data = d
        print(dialogue_id)
        frame, acoustic_data, lexical_data, pos_data, _, tag_labels = \
            dialogue_data
        if (not increco_style) and incremental_eval:
            acoustic_data, lexical_data, pos_data = \
                simulate_increco_data(frame, acoustic_data,
                                      lexical_data, pos_data)
            print("simulating asr")
        if hmm:
            hmm.viterbi_init()
        # 1. take increco input
        # 2. rollback if neccessary both the rnn state and MM state
        # 3. feed in the new hidden state and the saved MM state
        # 4. calculate the diff and write to increco style file with the timing
        if incremental_eval:
            softmaxlist = []
            history = []  # a the last 20 steps, can be edited
            current_words = []  # keep track of the list of words used
            current_words_string = []  # the words as strings
            current_pos = []
            started_results = False
            # these lists will act like increco results
            # i.e. new_lex list triples of (word hyp, start, end)
            # i.e. [('hello',0.2, 0.8), (...)... ]
            for new_lex, new_pos, new_acoustic in zip(lexical_data, pos_data,
                                                      acoustic_data):
                rollback = 0
                current_words, newsuffix, rollback = get_diff_and_new_prefix(
                    current_words, new_lex, verbose=False)
                current_pos = current_pos[:len(current_pos)-rollback] + new_pos
                current_words_string = current_words_string[
                    :len(current_words_string)-rollback] + \
                    [idx_to_word_dict[x[0][-1]] for x in new_lex]
                assert(len(current_words) == len(current_words_string) ==
                       len(current_pos))
                # history = history[:len(history)-rollback]
                # TODO was the history being reset incorrectly?
                history = history[:len(history)-rollback]
                softmaxlist = softmaxlist[:len(softmaxlist)-rollback]
                for i in range(max([len(softmaxlist), 0]), len(current_words)):
                    if not history == []:
                        if s["model"] == "lstm":
                            model.load_weights(c0=history[-1][0][-1],
                                               h0=history[-1][1][-1])
                        elif s["model"] == "elman":
                            model.load_weights(h0=history[-1][-1])
                        else:
                            raise NotImplementedError("no history loading for\
                                             {0} model".format(s["model"]))
                    l = current_words[i]
                    p = current_pos[i]
                    last_n_timings = get_last_n_features("timing",
                                                         current_words, i,
                                                         n=3)
                    if s["model"] == "lstm":
                        h_t, c_t, s_t = model.\
                            soft_max_return_hidden_layer([l[0]], [p])
                        softmaxlist.append(s_t)
                        if len(history) == 20:
                            history.pop(0)  # pop first one
                        history.append((c_t, h_t))
                    else:
                        h_t, s_t = model.soft_max_return_hidden_layer([l[0]],
                                                                      [p])
                        softmaxlist.append(s_t)
                        if len(history) == 20:
                            history.pop(0)  # pop first one
                        history.append(h_t)
                    softmax = np.concatenate(softmaxlist)
                    if not started_results:
                        started_results = True
                        file_id = id
                    else:
                        file_id = ""
                    row = softmax[i]  # should be just the right frontier
                    prediction_int = int(np.argmax(row))
                    raw_prediction = idx_to_label_dict[prediction_int]
                    raw_tag_predictions.append(raw_prediction)
                    if i < len(tag_labels):
                        label_idx = 0
                        raw_tag_labels.append(label_idx)
                        total += 1
                        nll = -log(row[label_idx])
                        loss += nll
                        class_loss[label_idx].append(nll)
                        # get the actual label
                        label = idx_to_label_dict[label_idx]
                        labels.append(label)
                    new_words = current_words_string[:i+1]
                    if hmm:
                        # if hmm get the proper viterbi prefix (or
                        # at least the changed indices for effiency
                        # else get all the predictions from the hmm increment
                        if rollback > 0:
                            hmm.rollback(rollback)
                            predictions = predictions[:-rollback]
                        rollback = 0
                        # after been rolled back once no need to do so again
                        adjustsoftmax = np.concatenate((
                            softmax,
                            softmax[:, label_to_idx_dict["<e/><cc>"]].
                            reshape(softmax.shape[0], 1)), 1)
                        newpredictions = hmm.viterbi_incremental(
                            adjustsoftmax, a_range=(i, i + 1),
                            changed_suffix_only=True,
                            timing_data=last_n_timings)
                        predictions = predictions[:len(predictions) -
                                                  (len(newpredictions) -
                                                  1)] + newpredictions
                        if "simple" not in s['tags']:
                            predictions = \
                                    convertFromIncDisfluencyTagsToEvalTags(
                                        predictions,
                                        new_words,
                                        start=len(predictions) -
                                        (len(newpredictions)),
                                        representation=s["tags"])
                        else:
                            for p in range(
                                    len(predictions) -
                                    (len(newpredictions) + 1),
                                    len(predictions)):
                                rps = predictions[p]
                                predictions[p] = rps.replace(
                                    'rm-0',
                                    'rps id="{}"'.format(p)).replace("<i",
                                                                     "<e/><i")
                    else:
                        prediction = idx_to_label_dict[prediction_int]
                        predictions.append(prediction)
                        predictions = convertFromIncDisfluencyTagsToEvalTags(
                            predictions,
                            new_words,
                            start=i,
                            representation=s["tags"])
                    if incremental_eval:
                        if file_id != "":
                            inc_predictions_file.write("\nFile: " +
                                                       str(file_id) + "\n")
                        inc_predictions_file.write('\nTime: ' +
                                                   str(current_words[-1][2]) +
                                                   "\n")
                        new_prefix = predictions[(len(prev_predictions)):]
                        for j in range(0, min([len(predictions),
                                               len(prev_predictions)])):
                            if predictions[j] != prev_predictions[j]:
                                new_prefix = predictions[j:]
                                break
                        for t, w, l in zip(current_words[
                            (i-(len(new_prefix)-1)):i+1],
                                        new_words[(i-(len(new_prefix)-1)):i+1],
                                        new_prefix):
                            _, start_time, end_time = t
                            inc_predictions_file.write("\t".join([
                                str(start_time), str(end_time), w,
                                str(l)])+"\n")
                        prev_predictions = deepcopy(predictions)
        else:  # not incremental eval, do something much easier to get the loss
            softmax = model.soft_max(lexical_data, pos_data)  # no acoustic
            for i in range(0, len(softmax)):
                row = softmax[i]  # should be just the right frontier
                prediction_idx = int(np.argmax(row))
                raw_tag_predictions.append(prediction_idx)
                if i < len(tag_labels):
                    label_idx = int(tag_labels[i])
                    raw_tag_labels.append(label_idx)
                    total += 1
                    nll = -log(row[label_idx])
                    loss += nll
                    class_loss[label_idx].append(nll)
    raw_predictions_file.close()
    if incremental_eval:
        inc_predictions_file.close()
    print("overall loss/cross entropy", loss/total)
    av_loss = []
    for key, val in sorted(list(class_loss.items()), key=lambda x: x[0]):
        print(key, val)
        av_loss.append(np.average(val))
    class_loss = np.average(av_loss)
    print("per class loss", class_loss)
    if incremental_eval:
        return {}
    p_r_f_tags = precision_recall_fscore_support(raw_tag_labels,
                                                 raw_tag_predictions,
                                                 average='weighted')
    tag_summary = classification_report(
        raw_tag_labels, raw_tag_predictions,
        labels=[i for i in range(len(classes))],
        target_names=[idx_to_label_dict[i] for i in range(len(classes))])
    print(tag_summary)
    results = {"f1_rmtto": p_r_f_tags[2], "f1_rm": p_r_f_tags[2],
               "f1_tto1": p_r_f_tags[2], "f1_tto2": p_r_f_tags[2]}

    results.update({
                'class_loss': class_loss,
                'loss': div(loss, total),
                'f1_tags': p_r_f_tags[2],
                'tag_summary': tag_summary
    })
    return results
