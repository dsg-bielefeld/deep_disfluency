"""Methods for disfluency and utterance segmentation from final and
increco-style inputs.
"""
from __future__ import division
import re
import numpy
from copy import deepcopy
from collections import defaultdict
import pandas as pd

# requires mumodo from https://github.com/dsg-bielefeld/mumodo
# from mumodo.analysis import intervalframe_overlaps
# from mumodo.plotting import plot_annotations


def intervalframe_overlaps(frame1, frame2, concatdelimiter='/'):
    """Modification from mumodo https://github.com/dsg-bielefeld/mumodo

    Intersection of two interval frames
    Return an IntervalFrame with the intersection  of two intervalframes
    An intersection is defined as an AND function on the intervals of both
    sources (regardless of text). HINT: Input Intervalframes should
    be imported without empty intervals
    Arguments:
    frame1,frame2   -- IntervalFrames.
    Keyword arguments:
    concatdelimiter  --  Concatenate the labels of the overlapping intervals
                         to create labels of the new dataframe intervals.
                         If empty string is given, the intervals are simply
                         labeled with 'overlap' instead.
    """
    overlaps = []
    if len(frame2) < len(frame1):
        frame1, frame2 = frame2, frame1
    for intrv1 in frame1.index:
        st1 = frame1['start_time'].ix[intrv1]
        en1 = frame1['end_time'].ix[intrv1]
        fr2 = frame2[frame2['end_time'] > st1]
        fr2 = fr2[fr2['start_time'] < en1]
        for intrv2 in fr2.index:
            overlap = {}
            if isinstance(concatdelimiter, str) and len(concatdelimiter) > 0:
                overlap['text'] = fr2.ix[intrv2]['text'] + concatdelimiter + \
                    frame1.ix[intrv1]['text']

            else:
                overlap['text'] = 'overlap'
            st2 = fr2['start_time'].ix[intrv2]
            en2 = fr2['end_time'].ix[intrv2]
            if st2 > st1:
                overlap['start_time'] = st2
            else:
                overlap['start_time'] = st1
            if en2 > en1:
                overlap['end_time'] = en1
            else:
                overlap['end_time'] = en2
            overlaps.append(overlap)
    return pd.DataFrame(overlaps).ix[:, ['start_time', 'end_time', 'text']]


def fill_in_time_approximations_timings(word_timing_tuples, idx):
    """Called when the start and end time of a word are the same
    or the start time is greater than the end time.
    (due to technical error), it searches forward
    in the list until
    it meets a word with a different end time, then spreads the time of
    the affected timing tuples stand and end time accordingly.

    word_timing_tuples :: list, of (word, start_time, end_time) triples
    idx :: int, the index of the first offending timing in the list
    """
    affected_indices = []
    start_time = word_timing_tuples[idx][0]
    end_time = word_timing_tuples[idx][1]
    assert start_time >= end_time,\
        "Function not needed- start and end times {0} and {1}".format(
            start_time, end_time)
    if start_time > end_time:
        end_time = start_time
    assert len(word_timing_tuples) > 1
    if end_time == word_timing_tuples[-1][1]:
        # this end time is the same as the latest one
        # need backward search for time
        # print "backward search"
        idx = len(word_timing_tuples) - 1
        for i in range(len(word_timing_tuples) - 1, -1, -1):
            affected_indices.append(i)
            idx = i
            if not word_timing_tuples[i][1] == start_time:
                start_time = word_timing_tuples[i][0]
                break
    else:
        for i in range(idx, len(word_timing_tuples)):
            affected_indices.append(i)
            if not word_timing_tuples[i][1] == start_time:
                end_time = word_timing_tuples[i][1]
                break
    total_time = end_time - start_time
    assert total_time > 0.0, str(word_timing_tuples[affected_indices[0]:]) +\
        str(idx)
    mean_len = total_time / len(affected_indices)
    for i in range(idx, idx + len(affected_indices)):
        end_time = start_time + mean_len
        word_timing_tuples[i] = (start_time,
                                 end_time)
        start_time = end_time
    return word_timing_tuples


def fill_in_time_approximations(word_timing_tuples, idx):
    """Called when the start and end time of a word are the same
    or the start time is greater than the end time.
    (due to technical error), it searches forward
    in the list until
    it meets a word with a different end time, then spreads the time of
    the affected timing tuples stand and end time accordingly.

    word_timing_tuples :: list, of (word, start_time, end_time) triples
    idx :: int, the index of the first offending timing in the list
    """
    start_time = word_timing_tuples[idx][1]
    end_time = word_timing_tuples[idx][2]
    assert start_time >= end_time,\
        "Function not needed- start and end times {0} and {1}".format(
            start_time, end_time)
    if start_time > end_time:
        end_time = start_time
    assert len(word_timing_tuples) > 1
    timings = [(x[1], x[2]) for x in word_timing_tuples]
    timings = fill_in_time_approximations_timings(timings, idx)
    for i, timing in zip(range(len(word_timing_tuples) - len(timings) - 1,
                               len(word_timing_tuples)),
                         timings):
        word_timing_tuples[i] = (word_timing_tuples[i][0],
                                 timing[0], timing[1])
    return word_timing_tuples


# IO methods for the different file types
def load_incremental_outputs_from_increco_file(increco_filename):
    """Loads increco style data from file.
    For now returns word, timing and tag data only
    """
    all_speakers = defaultdict(list)
    lex_data = []
    tag_data = []
    frames = []
    latest_increco = []
    latest_tag = []
    f = open(increco_filename)
    started = False
    conv_no = ""
    for line in f:
        # print line
        if "Time:" in line:
            if not latest_increco == []:
                lex_data.append(deepcopy(latest_increco))
                tag_data.append(deepcopy(latest_tag))
            latest_increco = []
            latest_tag = []
            continue
        if "Speaker:" in line:
            if not started:
                started = True
            else:
                # flush
                if not latest_increco == []:
                    lex_data.append(deepcopy(latest_increco))
                    tag_data.append(deepcopy(latest_tag))
                # print lex_data
                frames = [x[-1][-1] for x in lex_data]  # last word end time
                assert(len(frames) == len(lex_data) == len(tag_data)), \
                    "wrong lengths! {0} {1} {2}".format(len(frames),
                                                        len(lex_data),
                                                        len(tag_data))
                all_speakers[conv_no] = [deepcopy(frames),
                                         deepcopy(lex_data),
                                         deepcopy(tag_data)]
                # reset
                lex_data = []
                tag_data = []
                latest_increco = []
                latest_tag = []
            conv_no = line.strip("\n").replace("Speaker: ", "")
            continue
        if line.strip("\n") == "":
            continue
        spl = line.strip("\n").split("\t")
        start = float(spl[0])
        end = float(spl[1])
        assert len(spl) >= 3, "short line!" + "%".join(spl)
        # assert len(spl[2].split('@'))==2, "short 3rd idx!" + "%".join(spl)
        word = spl[2].split("@")[-1]
        tag = spl[-1].split("@")[-1]  # covers the POS cases
        latest_increco.append((word, start, end))
        latest_tag.append(tag)
    # flush
    if not latest_increco == []:
        lex_data.append(latest_increco)
        tag_data.append(latest_tag)
    frames = [x[-1][-1] for x in lex_data]  # last word end time
    all_speakers[conv_no] = [deepcopy(frames), deepcopy(lex_data),
                             deepcopy(tag_data)]
    assert(len(frames) == len(lex_data) == len(tag_data)), \
        "wrong lengths! {0} {1} {2}".format(len(frames),
                                            len(lex_data),
                                            len(tag_data))
    print len(all_speakers.keys()), "speakers"
    return all_speakers


def load_final_output_from_file(filename):
    """Just a generalization of the increco method whereby there
    is only one increco segment.
    """
    increco_speakers = load_incremental_outputs_from_increco_file(filename)
    final_dict = {}
    for speaker in increco_speakers.keys():
        assert(len(increco_speakers[speaker][0]) ==
               len(increco_speakers[speaker][1]) ==
               len(increco_speakers[speaker][2])), "lengths not the same!"\
            "\n" + str(len(increco_speakers[speaker][0])) +\
            " " + str(len(increco_speakers[speaker][1])) +\
            " " + str(len(increco_speakers[speaker][2])) + "\n" +\
            "\n".join(
            ['\t'.join([str(x), str(y), str(z)]) for x, y, z in zip(
                increco_speakers[speaker][0],
                increco_speakers[speaker][1],
                increco_speakers[speaker][2])]
        )
        final_timings = [(x[1], x[2]) for x in increco_speakers[speaker][1][0]]
        final_dict[speaker] = [[x[0] for x in final_timings],
                               [x[0] for x in increco_speakers[speaker][1][0]],
                               [(y, x[0], x[1]) for x, y in zip(final_timings,
                                                                increco_speakers[speaker][2][0])]]
    return final_dict


def add_word_continuation_tags(tags):
    """In place, add a continuation tag to each word:
    <cc/> -continues current utt and the next word will also continue it
    <ct/> -continues current utt and will end it
    <tc/> -starts a new utt and the next word will continue it
    <tt/> -starts and ends utt (single word dialogue act)
    """
    tags = list(tags)
    for i in range(0, len(tags)):
        if i == 0:
            tags[i] = tags[i] + "<t"
        else:
            tags[i] = tags[i] + "<c"
        if i == len(tags) - 1:
            tags[i] = tags[i] + "t/>"
        else:
            tags[i] = tags[i] + "c/>"
    return tuple(tags)


def get_diff_and_new_prefix(current, newprefix, verbose=False):
    """Only get the different right frontier according to the timings
    and change the current hypotheses.
    """
    if verbose:
        print "current", current
        print "newprefix", newprefix
    rollback = 0
    original_length = len(current)
    original_current = deepcopy(current)
    for i in range(len(current) - 1, -2, -1):
        if verbose:
            print "oooo", newprefix[0]
            if not current == []:
                print current[i]
        if i == -1 or (float(newprefix[0][1]) >= float(current[i][2])):
            if i == len(current) - 1:
                current = current + newprefix
                break
            k = 0
            marker = i + 1
            for j in range(i + 1, len(current)):
                if k == len(newprefix):
                    break
                if verbose:
                    print "...", j, k, current[j], newprefix[k], len(newprefix)
                if not current[j] == newprefix[k]:
                    break
                else:
                    if verbose:
                        print "repeat", current[j], newprefix[k]
                    k += 1
                    marker = j + 1
            rollback = original_length - marker
            current = current[:marker] + newprefix[k:]
            newprefix = newprefix[k:]
            break
    if newprefix == []:
        rollback = 0  # just no rollback if no prefix
        current = original_current  # reset the current
    rollback = min([original_length, rollback])
    if verbose:
        print "current after call", current
        print "newprefix after call", newprefix
        print "rollback after call", rollback
    return (current, newprefix, rollback)


def final_hyp_from_increco_and_incremental_metrics(increco,
                                                   gold,
                                                   goldwords,
                                                   utt_eval=False,
                                                   ttd_tags=None,
                                                   word=True,
                                                   interval=False,
                                                   tag_dict=None,
                                                   speaker_ID=None):
    """Returns the final sequence of each dialogue for non-incremental
    eval purposes. Also calculates the incremental metrics.
    """
    final_hypothesis = []
    final_words = []
    final_timings = []
    increco = deepcopy(increco)
    lengths = [len(increco[0]), len(increco[1]), len(increco[2])]
    if any([x != lengths[0] for x in lengths]):
        print len(increco[0]), len(increco[1]), len(increco[2])
        raw_input("problem0!")
    rollback = 0
    no_edits = 0  # for edit overhead (relative to final hyp)
    for _, n_words, n_tags in zip(increco[0], increco[1], increco[2]):
        assert len(n_words) == len(n_tags)
        new_tags = [(n_tags[x], n_words[x][1], n_words[x][2])
                    for x in range(0, len(n_words))]
        orig_length = len(final_hypothesis)
        final_hypothesis, new_prefix, rollback = \
            get_diff_and_new_prefix(deepcopy(final_hypothesis), new_tags,
                                    verbose=False)
        if len(new_prefix) < len(n_words):
            print "correcting length of words"
            n_words = n_words[(len(n_words) - len(new_prefix)):]
        assert len(n_words) == len(new_prefix)
        no_edits += len(new_prefix)
        final_words = final_words[:len(final_words) - rollback] \
            + deepcopy(n_words)
        # calculating incremental metrics
        for n in range(orig_length - rollback, len(final_hypothesis)):
            if not len(final_hypothesis[n]) == 3:
                print "uneven at number n!", n
                print final_hypothesis
            start_time = final_hypothesis[n][1]
            end_time = final_hypothesis[n][2]
            overlapped_intervals = [val for val in gold
                                    if val[1] >= start_time and
                                    val[2] <= end_time]
            for ttd_tag in ttd_tags:
                if ttd_tag in final_hypothesis[n][0]:
                    if word:
                        if not goldwords[n] == final_words[n][0] and \
                                not final_words[n][0] == "<unk>":
                            print "WARNING: different word hyp at index", n
                            count = 0
                            for x, y, w, z in zip(goldwords,
                                                  gold,
                                                  final_words,
                                                  final_hypothesis):
                                print count
                                count += 1
                                print x, y, w, z
                            print speaker_ID
                            raw_input()
                        if ttd_tag in gold[n][0]:
                            ttd_word = len(final_hypothesis) - n - 1
                            if "<r" in ttd_tag:
                                ttd_word += 1
                            tag_dict["t_t_detection_{0}_word"
                                     .format(ttd_tag)].append(ttd_word)
                    if interval:
                        if any([ttd_tag in x[0]
                                for x in overlapped_intervals]):
                            if ttd_tag in ["<rps", "<e"]:
                                ttd = float(final_hypothesis[-1][2] -
                                            overlapped_intervals[-1][1])
                            else:
                                ttd = float(final_hypothesis[-1][2] -
                                            overlapped_intervals[-1][2])
                            tag_dict["t_t_detection_{0}_interval"
                                     .format(ttd_tag)].append(ttd)
        if not len(final_words) == len(final_hypothesis):
            print len(final_timings), len(final_words), len(final_hypothesis)
            print final_words[-10:]
            print final_hypothesis[-10:]
            for w, h in zip(final_words, final_hypothesis):
                print w, h
            raw_input("problem1!")
    final_timings = [x[2] for x in final_words]
    if not len(final_timings) == len(final_words) == len(final_hypothesis):
        for t, w, h in zip(final_timings, final_words, final_hypothesis):
            print t, w, h
        raw_input("problem2 !")
    # incremental metrics update
    tag_dict["edit_overhead"][0] += no_edits
    tag_dict["edit_overhead"][1] += len(final_hypothesis)
    # final_words = [x[0] for x in final_words]  # change it to flat list
    return final_timings, final_words, final_hypothesis


def sort_into_dialogue_speakers(IDs, mappings, utts, pos_tags=None,
                                labels=None):
    """For each utterance, given its ID get its conversation number and
    dialogue participant in the format needed for word alignment files.

    Returns a list of tuples:
    (speaker, mappings, utts, pos, labels)

    """
    dialogue_speaker_dict = dict()  # keys are speaker IDs of filename:speaker
    # vals are tuples of (mappings, utts, pos_tags, labels)
    current_speaker = ""

    for ID, mapping, utt, pos, label in zip(IDs,
                                            mappings,
                                            utts,
                                            pos_tags,
                                            labels):
        split = ID.split(":")
        dialogue = split[0]
        speaker = split[1]
        # uttID = split[2]
        current_speaker = "".join([dialogue, speaker])
        if current_speaker not in dialogue_speaker_dict.keys():
            dialogue_speaker_dict[current_speaker] = [[], [], [], []]

        dialogue_speaker_dict[current_speaker][0].extend(list(mapping))
        dialogue_speaker_dict[current_speaker][1].extend(list(utt))
        dialogue_speaker_dict[current_speaker][2].extend(list(pos))
        dialogue_speaker_dict[current_speaker][3].extend(list(label))
    # turn into 5-tuples
    dialogue_speakers = [(key,
                          dialogue_speaker_dict[key][0],
                          dialogue_speaker_dict[key][1],
                          dialogue_speaker_dict[key][2],
                          dialogue_speaker_dict[key][3])
                         for key in sorted(dialogue_speaker_dict.keys())]
    return dialogue_speakers


def get_tag_data_from_corpus_file(f, representation="1", limit=8):
    """Loads from file into five lists of lists of strings of
    equal length:
        -utterance iDs (IDs))
        -word timings of the targets (start,stop)
        -words (seq),
        -POS tags(pos_seq)
        -labels (targets).

    NB this does not convert them into one-hot arrays,
    just outputs lists of string labels.
    """
    print "loading data", f
    f = open(f)
    count_seq = 0
    IDs = []
    seq = []
    pos_seq = []
    targets = []
    timings = []
    currentTimings = []

    counter = 0
    utt_reference = ""
    currentWords = []
    currentPOS = []
    currentTags = []

    for line in f:
        counter += 1
        if "Speaker:" in line:
            if count_seq > 0:  # do not reset the first time
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
            utt_reference = line.strip("\n").replace("Speaker: ", "")
        spl = line.strip("\n").split("\t")
        if not len(spl) == 6:
            continue
        _, start_time, end_time, word, postag, disftag = spl
        currentWords.append(word)
        currentPOS.append(postag)
        currentTags.append(disftag)
        approx_word_length = 0.3  # TODO approximation until end times gotten
        start = max([float(start_time), float(end_time) - approx_word_length])
        currentTimings.append((start, float(end_time)))
        if (len(currentTimings) > 1 and currentTimings[-2][1] > start) or \
                start >= float(end_time):
            # switch this start and end of the last word
            # start = currentTimings[-1][1]
            if (len(currentTimings) > 1 and currentTimings[-2][1] > start):
                currentTimings[-2] = (currentTimings[-2][0],
                                      currentTimings[-2][0])
                currentTimings = fill_in_time_approximations_timings(
                    currentTimings,
                    len(currentTimings) - 2)
            if currentTimings[-1][0] >= currentTimings[-1][1]:
                currentTimings[-1] = (currentTimings[-1][0],
                                      currentTimings[-1][0])
                currentTimings = fill_in_time_approximations_timings(
                    currentTimings,
                    len(currentTimings) - 1)
    if not currentWords == []:
        seq.append(tuple(currentWords))
        pos_seq.append(tuple(currentPOS))
        targets.append(tuple(currentTags))
        IDs.append(utt_reference)
        timings.append(tuple(currentTimings))
    assert len(seq) == len(targets) == len(pos_seq), ("{0} {1} {2}".format(
        len(seq),
        len(targets),
        len(pos_seq)
    )
    )
    print "loaded " + str(len(seq)) + " sequences"
    f.close()
    return (IDs, timings, seq, pos_seq, targets)


# Methods for general accuracy
def p_r_f(tps, fps, fns):
    if tps == 0:
        return (0, 0, 0)
    p = tps / (tps + fps)
    r = tps / (tps + fns)
    f = (2 * (p * r)) / (p + r)
    return (p, r, f)


# Methods for computing accuracy from dialogue final results
def NIST_SU(results):
    """Number of segmentation errors (missed segments and
    false alarm segments)
    over number of reference segments.
    """
    assert len(results) == 3
    TPs = results[0]
    FPs = results[1]
    FNs = results[2]
    if (FNs + FPs) == 0:
        return 0.0
    return ((FNs + FPs) / (TPs + FNs)) * 100


def DSER(results):
    """DA Segmentation Rate: number of segments of the
    reference incorrectly segmented
    over number of reference segments.
    """
    assert len(results) == 2
    CorrectSegs = results[0]
    TotalSegs = results[1]
    return ((TotalSegs - CorrectSegs) / TotalSegs) * 100


def SegER(results):
    """Segmentation Error Rate: edit distance between sequences of
    reference positions and the hypothesized positions
    """
    assert len(results) == 2
    Edits = results[0]
    TotalSegs = results[1]
    return (Edits / TotalSegs) * 100


def alignment_cost(r, h, subcost=1):
    """Calculation of Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    >>> alignment_cost("who is there".split(), "is there".split())
    1
    >>> alignment_cost("who is there".split(), "".split())
    3
    >>> alignment_cost("".split(), "who is there".split())
    3
    """
    # initialisation
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))

    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + subcost
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def final_output_accuracy_word_level(words, prediction_tags, gold_tags,
                                     tag_dict, utt_eval=False,
                                     error_analysis=False):
    """"Accuracy updates to the tag_dict for TPs, FPs, FNs.
    Assumes words, prediction_tags and gold_tags are the same length
    """
    relaxedGoldUtt, relaxedHypUtt = 0, 0
    repairs_hyp, repairs_gold, number_of_utts_hyp, number_of_utts_gold = \
        0, 0, 0, 0
    in_correct_segment = True
    assert len(words) == len(prediction_tags) == len(gold_tags), (
        "{0} {1} {2}".format(
            len(words),
            len(prediction_tags),
            len(gold_tags)
        ) +
        "\n%%%%%%words:" +
        "\n".join(words) + "\n%%%%%%prediction_tags:" +
        "\n".join(prediction_tags) +
        "\n%%%%%%\ngold_tags:" + "\n".join(gold_tags)
    )
    end_of_utt_align = {"ref": len(gold_tags) * [""],
                        "hyp": len(gold_tags) * [""]}
    count = 0
    for word, prediction, label in zip(words, prediction_tags, gold_tags):
        turnFinal = False
        if "<rps" in label:
            relaxedGoldUtt += 1
        if "<rps" in prediction:
            relaxedHypUtt += 1
        if "t/>" in label:
            number_of_utts_gold += 1
        for tag in tag_dict.keys():
            if tag in label and tag == "<rms":
                # special case for rms as possibly multiple
                gold_internal_tags = get_tags(label)
                gold_IDs = []
                for t in gold_internal_tags:
                    if "<rms" in t:
                        gold_ID = t[t.find("=") + 2:-3]
                        gold_IDs.append(gold_ID)
                # print "goldID", gold_ID
                # print "in", gold_tags
                gold_repairs = []
                gold_ID_dict = {k: False for k in gold_IDs}
                for i in range(count, len(gold_tags)):
                    for gold_ID in gold_IDs:
                        if '<rps id="{0}"/>'.format(gold_ID) in\
                                gold_tags[i]:
                            rps_gold_idx = i
                            gold_repairs.extend(
                                get_repairs_with_start_word_index(
                                    list(gold_tags), list(words),
                                    rps_gold_idx,
                                    gold_tags=list(gold_tags),
                                    save_context=True))
                            gold_ID_dict[gold_ID] = True
                    if all(gold_ID_dict.values()):
                        break

            if tag in prediction:
                correct_tag = False
                if tag == "<rms":  # special case for rms, possibly multiple
                    pred_internal_tags = get_tags(prediction)
                    pred_IDs = []
                    for t in pred_internal_tags:
                        if "<rms" in t:
                            pred_ID = t[t.find("=") + 2:-3]
                            pred_IDs.append(pred_ID)
                    # print "predID", pred_ID
                    # print "in", prediction_tags
                    # get all the repairs predicted which have the rms
                    # matching
                    pred_repairs = []
                    pred_ID_dict = {k: False for k in pred_IDs}
                    for i in range(count, len(prediction_tags)):
                        for pred_ID in pred_IDs:
                            if '<rps id="{0}"/>'.format(pred_ID) in\
                                    prediction_tags[i]:
                                rps_predicted_idx = i
                                pred_repairs.extend(
                                    get_repairs_with_start_word_index(
                                        list(prediction_tags), list(words),
                                        rps_predicted_idx,
                                        gold_tags=list(gold_tags),
                                        save_context=True))
                                pred_ID_dict[pred_ID] = True
                        if all(pred_ID_dict.values()):
                            break
                # see if there's a match for rms start
                if tag in label and tag == "<rms":
                    # Search for the corresponding rps tags
                    # rms is the one tag where multiple predictions
                    # and labels can be made for the same word
                    # at least one has to be correct
                    correct_rms_repairs = []
                    fp_rms_repairs = deepcopy(pred_repairs)

                    # generous match
                    # add all the correct and incorrect tags
                    for gr in gold_repairs:
                        # see if matching repair found for this gold repair
                        found = False
                        rps_found_indices = []
                        for pr_idx, pr in enumerate(fp_rms_repairs):
                            if gr.reparandumWords == pr.reparandumWords:
                                correct_rms_repairs.append(deepcopy(pr))
                                correct_tag = True
                                rps_found_indices.append(pr_idx)
                                found = True
                                break
                        for found_idx in rps_found_indices:
                            del fp_rms_repairs[found_idx]
                        if not found:
                            error_analysis[tag]["FN"].append(gr)
                            tag_dict[tag][2] += 1
                    if correct_tag and len(fp_rms_repairs) > 0:
                        # may have gotten one correct prediction for
                        # rms, but others wrong/FPs
                        error_analysis[tag]["FP"].extend(fp_rms_repairs)
                        tag_dict[tag][1] += (len(fp_rms_repairs))
                elif tag in label:
                    # any other tag, good if just matching label
                    correct_tag = True
                if correct_tag:
                    tag_dict[tag][0] += 1  # TPs
                    if error_analysis and tag in ["<rps", "<e", "t/>", "<rms"]:
                        if word:
                            if tag == "<rps":
                                error_analysis[tag]["TP"].append(
                                    get_repairs_with_start_word_index(
                                        list(gold_tags), list(words),
                                        count, gold_tags=list(gold_tags),
                                        save_context=True)[0]
                                )
                            elif tag == "<rms":
                                tag_dict[tag][0] += \
                                    (len(correct_rms_repairs) - 1)
                                error_analysis[tag]["TP"].extend(
                                    correct_rms_repairs
                                )
                            else:
                                error_analysis[tag]["TP"].extend(
                                    get_contexts_with_start_word_index(
                                        list(prediction_tags), list(words),
                                        count, gold_tags=list(gold_tags))
                                )
                        else:
                            pass
                            print "No error analysis at interval level"
                    if utt_eval and tag == "t/>":
                        end_of_utt_align["hyp"][count] = "1"
                        end_of_utt_align["ref"][count] = "1"
                        tag_dict["NIST_SU"][0] += 1
                        tag_dict["DSER"][1] += 1
                        if in_correct_segment:
                            tag_dict["DSER"][0] += 1
                        in_correct_segment = True  # resets
                else:
                    tag_dict[tag][1] += 1  # FPs
                    if error_analysis and tag in ["<rps", "<e", "t/>", "<rms"]:
                        if word:
                            if tag == "<rps":
                                error_analysis[tag]["FP"].append(
                                    get_repairs_with_start_word_index(
                                        list(prediction_tags), list(words),
                                        count, gold_tags=None,
                                        save_context=True)[0])
                            elif tag == "<rms":
                                # either predicted incorrectly
                                # or no matching rps found
                                tag_dict[tag][1] += (len(pred_repairs) - 1)
                                error_analysis[tag]["FP"].extend(pred_repairs)
                            else:
                                error_analysis[tag]["FP"].extend(
                                    get_contexts_with_start_word_index(
                                        list(prediction_tags), list(words),
                                        count, gold_tags=None))
                        else:
                            pass
                            print "No error analysis at interval level"
                    if utt_eval and tag == "t/>":
                        end_of_utt_align["hyp"][count] = "1"
                        tag_dict["NIST_SU"][1] += 1
                        in_correct_segment = False
            elif tag in label:
                tag_dict[tag][2] += 1  # FNs
                if error_analysis and tag in ["<rps", "<e", "t/>", "<rms"]:
                    if word:
                        if tag == "<rps":
                            error_analysis[tag]["FN"].append(
                                get_repairs_with_start_word_index(
                                    list(gold_tags), list(words), count,
                                    gold_tags=list(gold_tags),
                                    save_context=True)[0])
                        elif tag == "<rms":
                            tag_dict[tag][2] += (len(gold_repairs) - 1)
                            error_analysis[tag]["FN"].extend(gold_repairs)
                        else:
                            error_analysis[tag]["FN"].extend(
                                get_contexts_with_start_word_index(
                                    list(prediction_tags), list(words), count,
                                    gold_tags=list(gold_tags)))
                    else:
                        pass  # no error analysis for interval level yet
                if utt_eval and tag == "t/>":
                    end_of_utt_align["ref"][count] = "1"
                    tag_dict["NIST_SU"][2] += 1
                    tag_dict["DSER"][1] += 1
                    in_correct_segment = False
        count += 1
        # approximating the rates per utterance for repairs
        if (not utt_eval and "t/>" in label) or \
                (utt_eval and "t/>" in prediction):
            turnFinal = True
            tag_dict["<rps_relaxed"][0] += min(relaxedHypUtt, relaxedGoldUtt)
            tag_dict["<rps_relaxed"][1] += max(0,
                                               relaxedHypUtt - relaxedGoldUtt)
            tag_dict["<rps_relaxed"][2] += max(0,
                                               relaxedGoldUtt - relaxedHypUtt)
            repairs_gold += relaxedGoldUtt
            repairs_hyp += relaxedHypUtt
            number_of_utts_hyp += 1
            relaxedGoldUtt = 0
            relaxedHypUtt = 0
    if utt_eval:
        cost = alignment_cost(end_of_utt_align["ref"], end_of_utt_align["hyp"])
        tag_dict["SegER"][0] += tag_dict["DSER"][1]  # number of segments
        tag_dict["SegER"][1] += cost  # number of edits
    if not turnFinal:  # flush
        tag_dict["<rps_relaxed"][0] += min(relaxedHypUtt, relaxedGoldUtt)
        tag_dict["<rps_relaxed"][1] += max(0, relaxedHypUtt - relaxedGoldUtt)
        tag_dict["<rps_relaxed"][2] += max(0, relaxedGoldUtt - relaxedHypUtt)

    return repairs_hyp, repairs_gold, number_of_utts_hyp, number_of_utts_gold


def final_output_accuracy_interval_level(hyp, reference, tag_dict,
                                         utt_eval=False,
                                         error_analysis=False, window=10):
    r = [(float(a[0]), float(a[1]), d) for a, _, _, d in
         zip(reference[0], reference[1], reference[2], reference[3])]
    h = [(float(x[1]), float(x[2]), x[0]) for x in hyp[2]]

    reference = pd.DataFrame(r, columns=['start_time', 'end_time', 'text'])
    hyp = pd.DataFrame(h, columns=['start_time', 'end_time', 'text'])
    # interval accuracy based on:
    # TPs = sum of durations of intervals with hyp label value
    # intersecting with gold label value
    # FPs = sum of durations of intervals with hyp label value
    # present and not interecting with gold label value
    # FNs = sum of durations of intervals with gold label value
    # not intersecting with a hyp label value

    windows = []
    final_time = reference["end_time"].iloc[-1]
    start = 0
    end = window
    while end < final_time:
        windows.append((start, end))
        start += window
        end += window
    windows.append((start, final_time))
    repairs_hyp, repairs_gold, number_of_utts_hyp, number_of_utts_gold = \
        0, 0, 0, 0
    for tag in tag_dict.keys():
        if "<" not in tag and ">" not in tag:
            continue
        if tag in ["<rps", "<e", "t/>"]:
            for s, e in windows:
                refslots = reference[(reference['start_time'] >= s) &
                                     (reference['end_time'] < e) &
                                     (reference["text"].str.contains(tag))]
                relaxedGold = refslots.shape[0]
                hypslots = hyp[(hyp['start_time'] >= s) &
                               (hyp['end_time'] < e) &
                               (hyp["text"].str.contains(tag))]
                relaxedHyp = hypslots.shape[0]
                tag_dict["{}_relaxed".format(tag)][0]\
                    += min(relaxedHyp, relaxedGold)
                tag_dict["{}_relaxed".format(tag)][1]\
                    += max(0, relaxedHyp - relaxedGold)
                tag_dict["{}_relaxed".format(tag)][2]\
                    += max(0, relaxedGold - relaxedHyp)
        # now an overlap based eval
        gold_intervals = deepcopy(reference[reference.text.str.contains(tag)])
        hyp_intervals = deepcopy(hyp[hyp.text.str.contains(tag)])
        overlaps = deepcopy(intervalframe_overlaps(gold_intervals,
                                                   hyp_intervals))
        if tag == "t/>":
            number_of_utts_gold += len(gold_intervals)
            number_of_utts_hyp += len(hyp_intervals)
            # Convert to points for hyp and tolerance interval for gold
            # The tolerance allowed for overlap for end of utt detection
            tolerance = 0.75
            hyp_intervals['start_time'] = hyp_intervals['end_time'].\
                apply(lambda x: x - (0.001 / 2))
            hyp_intervals['end__time'] = hyp_intervals['end_time'].\
                apply(lambda x: x + (0.001 / 2))
            gold_intervals['start_time'] = gold_intervals['end_time'].\
                apply(lambda x: x - (tolerance / 2))
            gold_intervals['end__time'] = gold_intervals['end_time'].\
                apply(lambda x: x + (tolerance / 2))
            overlaps = deepcopy(intervalframe_overlaps(gold_intervals,
                                                       hyp_intervals))
            correctly_segmented_dser = 0
            in_correct_segment = True
            prev_end = -1
            for t in range(0, len(gold_intervals)):
                s = gold_intervals["start_time"].iloc[t]
                e = gold_intervals["end_time"].iloc[t]
                # check for intervening bad interval
                if hyp_intervals[(hyp_intervals['start_time'] >= prev_end) &
                                 (hyp_intervals['end_time'] <= s)].\
                        shape[0] > 0:
                    in_correct_segment = False
                if hyp_intervals[(hyp_intervals['start_time'] >= s) &
                                 (hyp_intervals['end_time'] <= e)].\
                        shape[0] > 0:
                    if in_correct_segment:
                        correctly_segmented_dser += 1
                    in_correct_segment = True
                else:
                    in_correct_segment = False
                prev_end = e
            total_hyp_segments = hyp_intervals.shape[0]
            correctly_segmented = overlaps.shape[0]
            total_gold_segments = gold_intervals.shape[0]
            tag_dict['DSER'][1] += total_gold_segments
            tag_dict['DSER'][0] += correctly_segmented_dser
            tag_dict['NIST_SU'][0] += correctly_segmented
            tag_dict['NIST_SU'][1] += (total_hyp_segments -
                                       correctly_segmented)
            tag_dict['NIST_SU'][2] += (total_gold_segments -
                                       correctly_segmented)
            continue
        gold_duration = sum(
            gold_intervals.end_time -
            gold_intervals.start_time)
        hyp_duration = sum(hyp_intervals.end_time - hyp_intervals.start_time)
        overlap_duration = sum(overlaps.end_time - overlaps.start_time)
        # do overall rate
        if tag == "<rps":
            repairs_hyp += len(hyp_intervals)
            repairs_gold += len(gold_intervals)
        if overlap_duration > gold_duration:
            for x in gold_intervals.index:
                print gold_intervals['start_time'][x] + " " +\
                    gold_intervals['end_time'][x] + " " +\
                    gold_intervals['text'][x]
                assert gold_intervals['start_time'][x]\
                    <= gold_intervals['end_time'][x]
            print hyp_intervals
            print overlap_duration, gold_duration, hyp_duration
            return False
        # TPs
        tag_dict[tag][0] += overlap_duration
        # FPs
        tag_dict[tag][1] += (hyp_duration - overlap_duration)
        # FNs
        tag_dict[tag][2] += (gold_duration - overlap_duration)
    return repairs_hyp, repairs_gold, number_of_utts_hyp, number_of_utts_gold


# Methods for error analysis on final results
class Context:
    """ A simple class which gives the words either side of a boundary word.
    """

    def __init__(self):
        self.words_left_context = []
        self.words_right_context = []
        self.tags_left_context = []
        self.tags_right_context = []
        self.gold_tags_left_context = []
        self.gold_tags_right_context = []
        self.type = None

    def __eq__(self, other):
        if self.left_context_words != other.left_context_words:
            return False
        if self.right_context_words != other.right_context_words:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        string = ""
        string += "left context=" + " ".join(["{0}|{1}".format(x, y)
                                              for x, y in zip(
            self.words_left_context,
            self.tags_left_context)]) + "\n"
        string += "right context=" + " ".join(["{0}|{1}".format(x, y)
                                               for x, y in zip(
            self.words_right_context,
            self.tags_right_context)]) \
            + "\n"
        string += "gold left context=" + " ".join(["{0}|{1}".format(x, y)
                                                   for x, y in
                                                   zip(
            self.words_left_context,
            self.gold_tags_left_context)])\
            + "\n"
        string += "gold right context=" + " ".join(["{0}|{1}".format(x, y)
                                                    for x, y in zip(
            self.words_right_context,
            self.gold_tags_right_context)])\
            + "\n"
        string += "type = " + str(self.type)
        return string


class Repair:
    """A simple class which gives the part of structure of the repair
    currently being consumed when queried.
    """

    def __init__(self):
        self.reparandumWords = []
        self.interregnumWords = []
        self.repairWords = []
        self.context = []
        self.gold_context = []
        self.reparandumComplete = False
        self.repairComplete = False
        self.type = None

    def __eq__(self, other):
        if self.reparandumWords != other.reparandumWords:
            return False
        if self.interregnumWords != other.interregnumWords:
            return False
        if self.repairWords != other.repairWords:
            return False
        if self.type != other.type:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        string = ""
        string += "reparandum = " + str(" ".join(self.reparandumWords)) + "\n"
        string += "interegnum = " + str(" ".join(self.interregnumWords)) + "\n"
        string += "repair = " + str(" ".join(self.repairWords)) + "\n"
        string += "context = " + str(" ".join(self.context)) + "\n"
        string += "gold_context = " + str(" ".join(self.gold_context)) + "\n"
        string += "type = " + str(self.type)
        return string


def get_tags(s, open_delim='<', close_delim='/>'):
    """Iterator to spit out the xml style disfluency tags in a
    given string.

    Keyword arguments:
    s -- input string
    """
    while True:
        # Search for the next two delimiters in the source text
        start = s.find(open_delim)
        end = s.find(close_delim)
        # We found a non-empty match
        if -1 < start < end:
            # Skip the length of the open delimiter
            start += len(open_delim)
            # Spit out the tag
            yield open_delim + s[start:end].strip() + close_delim
            # Truncate string to start from last match
            s = s[end + len(close_delim):]
        else:
            return


def get_contexts_with_start_word_index(tags, words, rp_start_index,
                                       gold_tags=None, window=5):
    """Returns a Context objects with the appropriate word structure.
    Only repairs with onset at index rp_start.
    """
    c = Context()
    c.words_left_context = words[rp_start_index - window: rp_start_index]
    c.words_right_context = words[rp_start_index: rp_start_index + window]
    c.tags_left_context = tags[rp_start_index - window: rp_start_index]
    c.tags_right_context = tags[rp_start_index: rp_start_index + window]
    if gold_tags:
        c.gold_tags_left_context = gold_tags[rp_start_index - window:
                                             rp_start_index]
        c.gold_tags_right_context = gold_tags[rp_start_index:
                                              rp_start_index + window]
    return [c]


def get_repairs_with_start_word_index(a_tags, a_words, rp_start_index,
                                      gold_tags=None, save_context=False,
                                      context_length=10):
    """Returns a list of Repair objects with the appropriate
    word structure.
    Only repairs with onset at index rp_start.
    """
    repairOnsets = re.findall('<rps id="[FP]*-?[0-9]*"/>',
                              a_tags[rp_start_index])
    repairDict = {rps[rps.find("=") + 2:-3]: None
                  for rps in repairOnsets}
    # first, backwards search for the "rms" tag
    allComplete = True
    if save_context:
        for repair_id in repairDict.keys():
            r = Repair()
            word_context = a_words[rp_start_index - context_length:
                                   rp_start_index] +\
                ["+"] + a_words[rp_start_index: rp_start_index + context_length]
            tag_context = a_tags[rp_start_index - context_length:
                                 rp_start_index] + ["+"] + \
                a_tags[rp_start_index: rp_start_index + context_length]
            r.context = ["{0}|{1}".format(x, y) for x, y in
                         zip(word_context, tag_context)]
            if gold_tags:
                tag_context = gold_tags[rp_start_index - context_length:
                                        rp_start_index] +\
                    ["+"] + gold_tags[rp_start_index:
                                      rp_start_index + context_length]
                r.gold_context = ["{0}|{1}".format(x, y) for x, y in
                                  zip(word_context, tag_context)]
            repairDict[repair_id] = deepcopy(r)
            # TODO assuming this is for false pos's only
    for c in range(rp_start_index - 1, -1, -1):
        allComplete = True
        for tag in get_tags(a_tags[c]):
            for repair_id in repairDict.keys():
                if repairDict[repair_id].reparandumComplete:
                    continue
                allComplete = False
                if '<i id="{}"/>'.format(repair_id) in tag:
                    repairDict[repair_id].interregnumWords.insert(0,
                                                                  a_words[c])
                if '<rm id="{}"/>'.format(repair_id) in tag:
                    repairDict[repair_id].reparandumWords.insert(0,
                                                                 a_words[c])
                if '<rms id="{}"/>'.format(repair_id) in tag:
                    repairDict[repair_id].reparandumWords.insert(0,
                                                                 a_words[c])
                    repairDict[repair_id].repararandumComplete = True
        if allComplete:
            break
    # now forwards search for the rpn tag and type
    allComplete = True
    for c in range(rp_start_index, len(a_tags)):
        allComplete = True
        for repair_id in repairDict.keys():
            wordAdded = False
            for tag in get_tags(a_tags[c]):
                if repairDict[repair_id].repairComplete:
                    continue
                allComplete = False
                if re.match('<rp[s] id="{}"/>'.format(repair_id), tag):
                    if not wordAdded:
                        repairDict[repair_id].repairWords.append(a_words[c])
                    wordAdded = True
                if re.match('<rpn[repdelsub]* id="{}"/>'.format(repair_id),
                            tag):
                    if not wordAdded:
                        repairDict[repair_id].repairWords.append(a_words[c])
                    wordAdded = True
                    # sometimes there is a rpn and rpndel, take the del
                    if ("<rpn " in tag) and (not
                                             repairDict[repair_id].repairComplete) and \
                            '<rpndel id="{}"/>'.format(repair_id) in a_tags[c]:
                        continue
                    repairDict[repair_id].repairComplete = True
                    if "<rpn " in tag:
                        # may miss off official repair categorization
                        if repairDict[repair_id].reparandumWords == \
                                repairDict[repair_id].repairWords:
                            repairDict[repair_id].type = "rep"
                        else:
                            repairDict[repair_id].type = "sub"
                    else:
                        repairDict[repair_id].type = tag[4:7]
        if allComplete:
            break
    if len(repairDict.values()) == 0:
        print "warning, no repairs found at ", rp_start_index
        for i, tag in enumerate(a_tags):
            print i, tag
    return deepcopy([val for val in repairDict.values()])


def rename_all_repairs_in_line_with_index(tags, simple=False):
    """Ensure all repairs are renamed according to the index
    their repair onset word starts at.
    """
    for i in range(0, len(tags)):
        for tag in get_tags(tags[i]):
            if "<rps" in tag:
                repairOnsets = re.findall('<rps id="[FP]*-?[0-9]*"/>', tags[i])
                assert len(repairOnsets) == 1, "too many/few repairs at" + \
                    str(i) + " in \n " + \
                    "\n".join(["\t".join([str(x), str(y)]) for
                               x, y in zip(range(0, len(tags)), tags)])
                rps = repairOnsets[0]
                old_id = rps[rps.find("=") + 2:-3]
                if old_id != str(i):
                    tags = rename_repair_with_repair_onset_idx(tags,
                                                               i,
                                                               str(i),
                                                               simple=simple)
    return tags


def rename_repair_with_repair_onset_idx(orig_tags, rp_start_index, new_id,
                                        simple=False,
                                        verbose=False):
    """Returns tags with the repair with a repair onset beginning\
    at position repair_start_idx renamed to string new_id"""
    tags = deepcopy(orig_tags)
    repairOnsets = re.findall('<rps id="[FP]*-?[0-9]*"/>',
                              tags[rp_start_index])
    repairDict = {rps[rps.find("=") + 2:-3]: None
                  for rps in repairOnsets}
    assert len(repairOnsets) == 1, "too many/few repairs at" + \
        tags[rp_start_index]
    # from the repair onset go back and replace the relevant tag
    rps = repairOnsets[0]
    old_id = rps[rps.find("=") + 2:-3]
    repair_start_found = False
    for i in range(rp_start_index - 1, -1, -1):
        new_tag = ""
        for tag in get_tags(tags[i]):
            if tag == '<i id="{}"/>'.format(old_id):
                new_tag += '<i id="{}"/>'.format(new_id)
            elif tag == '<rm id="{}"/>'.format(old_id):
                new_tag += '<rm id="{}"/>'.format(new_id)
            elif tag == '<rms id="{}"/>'.format(old_id):
                new_tag += '<rms id="{}"/>'.format(new_id)
                repair_start_found = True
            else:
                new_tag += tag
        tags[i] = new_tag
        if repair_start_found:
            break
    if (not simple) and (not repair_start_found):
        raise Exception("No start found for repair beginning at " +
                        str(rp_start_index) + " in: \n" +
                        "\n".join(["\t".join([str(x), str(y)]) for
                                   x, y in zip(range(0, len(orig_tags)),
                                               orig_tags)]))
    repair_end_found = False
    furthest_point = rp_start_index
    found_rps = False
    same_name_repair_found = False
    for i in range(rp_start_index, len(tags)):
        new_tag = ""
        for tag in get_tags(tags[i]):
            if tag == '<rps id="{}"/>'.format(old_id):
                if found_rps:
                    same_name_repair_found = True
                    new_tag = tag
                    break
                new_tag += '<rps id="{}"/>'.format(new_id)
            elif tag == '<rp id="{}"/>'.format(old_id):
                new_tag += '<rp id="{}"/>'.format(new_id)
                furthest_point = i
            elif tag == '<rpnsub id="{}"/>'.format(old_id):
                new_tag += '<rpnsub id="{}"/>'.format(new_id)
                repair_end_found = True
            elif tag == '<rpnrep id="{}"/>'.format(old_id):
                new_tag += '<rpnrep id="{}"/>'.format(new_id)
                repair_end_found = True
            elif tag == '<rpndel id="{}"/>'.format(old_id):
                new_tag += '<rpndel id="{}"/>'.format(new_id)
                repair_end_found = True
            elif tag == '<rpn id="{}"/>'.format(old_id):
                new_tag += '<rpn id="{}"/>'.format(new_id)
                repair_end_found = True
            else:
                new_tag += tag
        tags[i] = new_tag
        if repair_end_found or same_name_repair_found:
            break
    if (not simple) and (not repair_end_found):
        if verbose:
            print("No end found for repair beginning at " +
                  str(rp_start_index) + " in: \n" +
                  "\n".join(["\t".join([str(x), str(y)]) for
                             x, y in zip(range(0, len(orig_tags)),
                                         orig_tags)]))
            print "correcting end"
        tags[furthest_point] = tags[furthest_point].replace(
            '<rp id="{}"/>'.format(old_id), "") + \
            '<rpn id="{}"/>'.format(new_id)
    return tags


# Methods for error analysis on incremental results
def test():
    print "testing repair extraction"
    # tags = '<f/>,<rms id="FP3"/>,<i id="FP3"/><e/>,<rps id="FP3"/>\
    # <rpn id="FP3"/>,<rms id="6"/>,<i id="6"/><e/>,<rps id="6"/>\
    # <rpn id="6"/>,<f/>'.split(',')
    tags = '<f/>,<f/>,<i id="FP3"/><e/>,<rps id="FP3"/>,<f/>,<i id="6"/><e/>,<rps id="6"/>\
    <f/>,<f/>'.split(',')
    print rename_repair_with_repair_onset_idx(tags, 3, "FP0",
                                              simple=True)

    print re.findall('<rps id="[FP]*-?[0-9]*"/>', '<rps id="-1"/>')

    words = "i,like,uh,love,like,uh,love,alot".split(",")
    repairs = get_repairs_with_start_word_index(tags, words, 3)
    for r in repairs:
        print r, len(r.reparandumWords)
    repairs2 = get_repairs_with_start_word_index(tags, words, 6)
    for r in repairs2:
        print r
    print repairs2[0] == repairs[0]


if __name__ == '__main__':
    test()
