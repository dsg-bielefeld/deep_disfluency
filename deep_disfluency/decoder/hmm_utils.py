from copy import deepcopy
import math
import nltk


def log(prob):
    if prob == 0.0:
        return - float("Inf")
    return math.log(prob, 2)


def tabulate_cfd(cfd, *args, **kwargs):
    """
    Tabulate the given samples from the conditional frequency distribution
    or conditional probability distribution.

    :param samples: The samples to plot
    :type samples: list
    :param conditions: The conditions to plot (default is all)
    :type conditions: list
    :param cumulative: A flag to specify whether the freqs are cumulative
    (default = False)
    :type title: bool
    """

    cumulative = False
    # conditions = sorted([c for c in cfd.conditions()
    #                     if "_t_" in c])   # only concerned with act-final
    conditions = sorted([c for c in cfd.conditions()])
    if type(cfd) == nltk.ConditionalProbDist:
        samples = sorted(set(v for c in conditions for v in cfd[c].samples()))
    else:
        samples = sorted(set(v for c in conditions for v in cfd[c]))
    if samples == []:
        print "No conditions for tabulate!"
        return None
    width = max(len("%s" % s) for s in samples)
    freqs = dict()
    for c in conditions:
        if cumulative:
            freqs[c] = list(cfd[c]._cumulative_frequencies(samples))
        else:
            if type(cfd) == nltk.ConditionalProbDist:
                freqs[c] = [cfd[c].prob(sample) for sample in samples]
            else:
                freqs[c] = [cfd[c][sample] for sample in samples]
        width = max(width, max(len("%d" % f) for f in freqs[c]))

    # condition_size = max(len("%s" % c) for c in conditions)
    final_string = ""
    # final_string += ' ' * condition_size
    if type(cfd) == nltk.ConditionalProbDist:
        width += 1
    for s in samples:
        # final_string += "%*s" % (width, s)
        final_string += "\t" + str(s)
    final_string = final_string + "\n"
    for c in conditions:
        # final_string += "%*s" % (condition_size, c)
        final_string += str(c) + "\t"
        for f in freqs[c]:

            if type(cfd) == nltk.ConditionalProbDist:
                # final_string += "%*.2f" % (width, f)
                if f == 0:
                    final_string += "\t"
                else:
                    final_string += "{:.3f}\t".format(f)
            else:
                # final_string += "%*d" % (width, f)
                if f == 0:
                    final_string += "\t"
                else:
                    final_string += "{}\t".format(f)
        final_string = final_string[:-1] + "\n"
    return final_string


def convert_to_dot(filename):
    """Converts the transition matrix shown in a csv file to a
    dot formatted graph.
    """
    csv_file = open(filename)
    lines = csv_file.readlines()
    header = lines[0].split('\t')[1:]  # header for the second one
    graph_string = ""
    for line in lines[1:]:
        feats = line.split('\t')
        domain = feats[0]
        for i in range(1, len(feats)):
            if feats[i].strip() == "1":
                graph_string += domain + " -> " + \
                    header[i - 1].strip().strip("\n") + ";\n"
    file.close()
    return graph_string


def fill_in_time_approximations(word_timing_tuples, idx):
    """Called when the start and end time of a word are the same
    (i.e. it has no length, due to technical error), it searches forward
    in the list until
    it meets a word with a different end time, then spreads the time of
    the affected timing tuples stand and end time accordingly.

    word_timing_tuples :: list, of (word, start_time, end_time) triples
    idx :: int, the index of the first offending timing in the list
    """
    affected_indices = []
    start_time = word_timing_tuples[idx][1]
    end_time = word_timing_tuples[idx][2]
    assert start_time == end_time,\
        "Function not needed- different start and end times!"
    assert len(word_timing_tuples) > 1
    if end_time == word_timing_tuples[-1][2]:
        # this end time is the same as the latest one
        # need backward search for time
        # print "backward search"
        idx = len(word_timing_tuples) - 1
        for i in range(len(word_timing_tuples) - 1, -1, -1):
            affected_indices.append(i)
            idx = i
            if not word_timing_tuples[i][1] == start_time:
                start_time = word_timing_tuples[i][1]
                break
    else:
        for i in range(idx, len(word_timing_tuples)):
            affected_indices.append(i)
            if not word_timing_tuples[i][2] == start_time:
                end_time = word_timing_tuples[i][2]
                break
    total_time = end_time - start_time
    assert total_time > 0.0, str(word_timing_tuples[affected_indices[0]:]) +\
        str(idx)
    mean_len = total_time / len(affected_indices)
    for i in range(idx, idx + len(affected_indices)):
        end_time = start_time + mean_len
        word_timing_tuples[i] = (word_timing_tuples[i][0], start_time,
                                 end_time)
        start_time = end_time
    return word_timing_tuples


def load_data_from_corpus_file(filename):
    """Loads from disfluency detection with timings file.
    """
    all_speakers = []
    lex_data = []
    pos_data = []
    frames = []
    labels = []

    latest_increco = []
    latest_pos = []
    latest_labels = []

    a_file = open(filename)
    started = False
    conv_no = ""
    prev_word = "<s/>"  # -1
    prev_pos = "<s/>"  # -1
    print "loading in timings file", filename, "..."
    for line in a_file:
        if "Speaker:" in line:
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
                        latest_increco = fill_in_time_approximations(
                            latest_increco,
                            shift)
                    lex_data.extend(deepcopy(latest_increco))
                    pos_data.extend(deepcopy(latest_pos))
                    # convert to the disfluency tags for this
                    # latest_labels = convertFromEvalTagsToIncDisfluencyTags()
                    labels.extend(deepcopy(latest_labels))
                # fake
                # print lex_data
                frames = [x[-1] for x in lex_data]  # last word end time
                # acoustic_data = [0] * len(lex_data)  # fakes
                indices = [0] * len(lex_data)

                all_speakers.append((conv_no, (frames, lex_data, pos_data,
                                               indices, labels)))
                # reset
                lex_data = []
                pos_data = []
                labels = []
                latest_increco = []
                latest_pos = []
                latest_labels = []
                prev_word = "<s/>"  # -1
                prev_pos = "<s/>"  # -1

            conv_no = line.strip("\n").replace("Speaker: ", "")
            continue
        if line.strip("\n") == "":
            continue
        spl = line.strip("\n").split("\t")
        # print "@@@@" + line + "@@@@"
        # raw_input()
        start = float(spl[1])
        end = float(spl[2])
        word = spl[3]
        pos = spl[4]
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
        lex_data.extend(latest_increco)
        pos_data.extend(latest_pos)
        labels.extend(latest_labels)
    frames = [x[-1] for x in lex_data]  # last word end time
    # acoustic_data = [0,] * len(lex_data)  # fakes..
    indices = [0] * len(lex_data)
    all_speakers.append((conv_no, (frames, lex_data, pos_data,
                                   indices, labels)))
    print len(all_speakers), "speakers with timings input"
    # print "first few lengths"
    # limit = 10
    # for s in all_speakers:
    #     id, data = s
    #     print [len(x) for x in data]
    #     limit-=1
    #     if limit<0: break
    return all_speakers
