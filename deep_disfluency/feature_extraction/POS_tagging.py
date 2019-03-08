
import os
import sys

from copy import deepcopy
from collections import Counter
from nltk.tag import CRFTagger

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../..")
from deep_disfluency.feature_extraction.feature_utils import\
    load_data_from_disfluency_corpus_file
from deep_disfluency.feature_extraction.feature_utils import\
     sort_into_dialogue_speakers


# Set the variables according to what you want to do
TRAIN = True
TEST = True
TAG_ASR_RESULTS = False

TAGGER_PATH = "crfpostagger"  # path to the tagger you want to train/apply

# Train and test from disfluency detection format files:
DISF_DIR = THIS_DIR + "/../data/disfluency_detection/switchboard"
DISFLUENCY_TRAIN_FILES = [
                    DISF_DIR + "/swbd_disf_train_1_partial_data.csv",
                    ]
DISFLUENCY_TEST_FILES = [
                    DISF_DIR + "/swbd_disf_heldout_partial_data.csv",
                    ]

# ASR results from increco outputs
ASR_DIR = THIS_DIR + "/../data/asr_results/"
INC_ASR_FILES = [ASR_DIR + "SWDisfTest_increco.text",
                 ASR_DIR + "SWDisfHeldout_increco.text"]

# The tags for which an entity tag is added to the word
PROPER_NAMES = ["NNP", "NNPS", "CD", "LS", "SYM", "FW"]

if TRAIN:
    ct = CRFTagger()  # initialize tagger
    dialogue_speakers = []
    for disf_file in DISFLUENCY_TRAIN_FILES:
        IDs, mappings, utts, pos_tags, labels = \
            load_data_from_disfluency_corpus_file(disf_file)
        dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,
                                                             mappings,
                                                             utts,
                                                             pos_tags,
                                                             labels))
    word_pos_data = {}  # map from the file name to the data
    for data in dialogue_speakers:
        dialogue, a, b, c, d = data
        word_pos_data[dialogue] = (a, b, c, d)

    training_data = []
    for speaker in list(word_pos_data.keys()):
        sp_data = []
        prefix = []
        for word, pos in zip(word_pos_data[speaker][1],
                             word_pos_data[speaker][2]):
            prefix.append(word.replace("$unc$", ""))
            sp_data.append((str(word.replace("$unc$", "")
                                    .encode("utf8")),
                            str(pos.encode("utf8"))))
        training_data.append(deepcopy(sp_data))
    print("training tagger...")
    ct.train(training_data, TAGGER_PATH)


if TEST:
    print("testing tagger...")
    ct = CRFTagger()  # initialize tagger
    ct.set_model_file(TAGGER_PATH)
    dialogue_speakers = []
    for disf_file in DISFLUENCY_TEST_FILES:
        IDs, mappings, utts, pos_tags, labels = \
            load_data_from_disfluency_corpus_file(disf_file)
        dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,
                                                             mappings,
                                                             utts,
                                                             pos_tags,
                                                             labels))
    word_pos_data = {}  # map from the file name to the data
    for data in dialogue_speakers:
        dialogue, a, b, c, d = data
        word_pos_data[dialogue] = (a, b, c, d)
    ct.tag([str(w) for w in "uh my name is john".split()])
    # either gather training data or test data
    training_data = []
    for speaker in list(word_pos_data.keys()):
        # print speaker
        sp_data = []
        prefix = []
        predictions = []
        for word, pos in zip(word_pos_data[speaker][1],
                             word_pos_data[speaker][2]):
            prefix.append(str(word.replace("$unc$", "")
                                  .encode("utf8")))
            prediction = ct.tag(prefix[-5:])[-1][1]
            sp_data.append((str(word.replace("$unc$", "")
                                    .encode("utf8")),
                            str(pos.encode("utf8"))))
            predictions.append(prediction)
        training_data.append(deepcopy([(r, h)
                                       for r, h in zip(predictions, sp_data)]))
    # testing
    tp = 0
    fn = 0
    fp = 0
    overall_tp = 0
    overall_count = 0
    c = Counter()
    for t in training_data:
        for h, r in t:
            # print h,r
            overall_count += 1
            hyp = h
            if hyp == "UH":
                if not r[1] == "UH":
                    fp += 1
                else:
                    # print h,r
                    tp += 1
            elif r[1] == "UH":
                fn += 1
            if hyp == r[1]:
                overall_tp += 1
            else:
                c[hyp + "-" + r[1]] += 1
    print(tp, fn, tp)
    p = (tp/(tp+fp))
    r = (tp/(tp+fn))
    print("UH p, r, f=", p, r, (2 * p * r)/(p+r))
    print("overall accuracy", overall_tp/overall_count)
    print("most common errors hyp-ref", c.most_common()[:20])

if TAG_ASR_RESULTS:
    def get_diff_and_new_prefix(current, newprefix, verbose=False):
        """Only get the different right frontier according to the timings
        and change the current hypotheses"""
        if verbose:
            print("current", current)
            print("newprefix", newprefix)
        rollback = 0
        original_length = len(current)
        for i in range(len(current)-1, -2, -1):
            if verbose:
                print("oooo", current[i], newprefix[0])
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
                        print("...", j, k, current[j], newprefix[k])
                        print(len(newprefix))
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
        return (current, newprefix, rollback)

    print("tagging ASR results...")
    ct = CRFTagger()  # initialize tagger
    ct.set_model_file(TAGGER_PATH)
    # now tag the incremental ASR result files of interest
    for filename in INC_ASR_FILES:
        # always tag the right frontier
        current = []
        right_frontier = 0
        rollback = 0
        newfile = open(filename.replace("increco.",
                                        "pos_increco."),
                       "w")
        a_file = open(filename)
        dialogue = 0
        for line in a_file:
            if "File:" in line:
                dialogue = line
                newfile.write(line)
                current = []
                right_frontier = 0
                rollback = 0
                continue
            if "Time:" in line:
                increment = []
                newfile.write(line)
                continue
            if line.strip("\n") == "":
                if current == []:
                    current = deepcopy(increment)
                else:
                    verb = False
                    current, _, rollback = get_diff_and_new_prefix(
                                                deepcopy(current),
                                                deepcopy(increment),
                                                verb)
                for i in range(right_frontier - rollback, len(current)):
                    test = [str(x[0].lower().replace("'", ""))
                            for x in current[max([i-4, 0]):i+1]]
                    # if "4074A" in dialogue:
                    #    print "test", test
                    prediction = ct.tag(test)[-1][1]
                    word = current[i][0].lower().replace("'", "")
                    if prediction in PROPER_NAMES:
                        word = "$unc$" + word
                    start = current[i][1]
                    end = current[i][2]
                    newfile.write("\t".join([str(start),
                                            str(end), word] +
                                            [prediction]) + "\n"
                                  )
                right_frontier = len(current)
                newfile.write(line)
            else:
                spl = line.strip("\n").split("\t")
                increment.append((spl[0], float(spl[1]), float(spl[2])))
        file.close()
        newfile.close()
