# To affect an HSMM, we want the probability of the state given the length of
# time to the last word
# This is only for non-0 transitions in the HMM


import numpy as np
import time
from collections import defaultdict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# import sys
# sys.path.append("../")

from deep_disfluency.feature_extraction.feature_utils import \
    load_data_from_corpus_file
from deep_disfluency.load.load import load_tags


def load_timing_data(dialogues, labels2idx, simple=False):
    # make the args into a dict
    # get relevant dictionaries

    if simple:
        # overall_simple_trp_label2idx = {0: "<cc/>",
        #                                 1: "<ct/>",
        #                                 2: "<tc/>",
        #                                 3: "<tt/>"}
        simple_trp_label2idx = {0: "<c",
                                1: "<t"}
        # simple_trp_label2idx2 = {0: "c/>",
        #                          1: "t/>"}
    else:
        simple_trp_label2idx = labels2idx
    # a map from the label index to every duration of that label
    timing_dict = defaultdict(list)
    for dialogue in dialogues:
        _, dialogue_speaker_data = dialogue
        _, lex_data, _, _, labels = dialogue_speaker_data
        timings = [float(l[2]) for l in lex_data]  # just word end timings
        prev = 0
        prev_t = 0
        prev_prev_t = 0  # effecting a trigram of timings
        for i in range(0, len(timings)):
            t = timings[i] - prev
            complex_tag = labels[i]
            # print complex_tag
            found = False
            tag = labels[i]
            if simple:
                for k, v in list(simple_trp_label2idx.items()):
                    if v in complex_tag:
                        tag = k
                        found = True
                        break
                if not found:
                    if "<laughter" in complex_tag:
                        continue
                    input("warning: " + complex_tag + " " + tag)
            if t < 0:
                print("below zero")
                t = np.average([x[0] for x in timing_dict[tag]])
                timings[i-1] = timings[i] - t
            # turn to milliseconds
            timing_dict[tag].append((prev_prev_t, prev_t,
                                     t))
            # now the second one
            prev = timings[i]
            prev_prev_t = prev_t
            prev_t = t

    # compile all the timings information together
    X = []
    y = []
    for i in sorted(timing_dict.keys()):
        print(simple_trp_label2idx[i])
        print(np.average([time[0] for time in timing_dict[i]]), end=' ')
        print(np.std([time[0] for time in timing_dict[i]]))
        for tup in timing_dict[i]:
            X.append(list(tup))
            y.append(i)
    X = np.asarray(X)
    y = np.asarray(y)
    print(X.shape)
    print(y.shape)
    return X, y


def train(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # quite a sparsity problem
    model = LogisticRegression(class_weight='balanced',
                               multi_class='ovr')
    model.fit(X, y)

    print(model)
    return model, scaler


def test_simple(model, scaler, data, y):
    # make predictions
    X = scaler.transform(data)
    expected = y
    print(expected)
    predicted = model.predict(X)
    print(model.predict_proba(X))
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


def test(model, scaler, X, y):
    # make predictions
    def convert_to_overall_dict(a, b):
        if a == 0:
            if b == 0:
                return 0
            elif b == 1:
                return 1
        elif a == 1:
            if b == 0:
                return 2
            elif b == 1:
                return 3
        print(a, b, "wrong!")
        return None

    def convert_to_two_singles(a):
        if a == 0:
            return 0, 0
        if a == 1:
            return 0, 1
        if a == 2:
            return 1, 0
        if a == 3:
            return 1, 1
    test1 = []
    for x in list(y):
        a, _ = convert_to_two_singles(x)
        test1.append(a)

    X1 = scaler.transform(X)

    predicted = model.predict(X1)
    print(metrics.classification_report(np.asarray(test1), predicted))
    print(metrics.confusion_matrix(np.asarray(test1), predicted))

if __name__ == '__main__':
    disf_dir = "../data/disfluency_detection/switchboard"
    training_file = disf_dir + "/swbd_disf_train_1_2_partial_data_timings.csv"
    heldout_file = disf_dir + "/swbd_disf_heldout_partial_data_timings.csv"

    tag_dir = "../data/tag_representations"
    labels2idx = load_tags(tag_dir + "/swbd_uttseg_tags.csv")
    dialogues = load_data_from_corpus_file(training_file)
    X, y = load_timing_data(dialogues, labels2idx, simple=True)
    model, scaler = train(X, y)

    dialogues = load_data_from_corpus_file(heldout_file)
    X, y = load_timing_data(dialogues, labels2idx, simple=True)
    test_simple(model, scaler, X, y)

    # save the classifier
    with open('timing_models/' +
              'LogReg_balanced_timing_classifier.pkl', 'wb') as fid:
        pickle.dump(model, fid)
    with open('timing_models/' +
              'LogReg_balanced_timing_scaler.pkl', 'wb') as fid:
        pickle.dump(scaler, fid)
