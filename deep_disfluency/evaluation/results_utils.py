import pandas as pd
from collections import OrderedDict
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from disf_evaluation import ACCURACY_HEADER
from disf_evaluation import FINAL_OUTPUT_TTO_ACCURACY_HEADER
from disf_evaluation import INCREMENTAL_OUTPUT_TTO_ACCURACY_HEADER


final_result_to_latex_dict = OrderedDict((key, val) for key, val in [
    ("f1_<rm_word", """$F_{rm}$ (per word)"""),
    ("f1_<rps_word", """$F_{rps}$ (per word)"""),
    ("f1_<e_word", """$F_{e}$ (per word)"""),
    ("f1_t/>_word",  """$F_{uttSeg}$ (per word)"""),
    ("f1_<rps_relaxed_interval", """$F_{rps}$ (per 10s window)"""),
    ("p_<rps_relaxed_interval", """$P_{rps}$ (per 10s window)"""),
    ("r_<rps_relaxed_interval", """$R_{rps}$ (per 10s window)"""),
    ("f1_<rps_relaxed_word", """$F_{rps}$ (per utterance)"""),
    ("p_<rps_relaxed_word", """$P_{rps}$ (per utterance)"""),
    ("r_<rps_relaxed_word", """$R_{rps}$ (per utterance)"""),
    ("f1_<e_relaxed_interval", """$F_{e}$ (per 10s window)"""),
    ("f1_t/>_relaxed_interval", """$F_{uttSeg}$ (per 10s window)"""),
    ("pearson_r_correl_rps_rate_per_utt",
     "$rps$ per utterance per speaker Pearson R correlation"),
    ("spearman_rank_correl_rps_rate_per_utt",
     "$rps$ per utterance per speaker Spearman's Rank correlation"),
    ("NIST_SU_word", "NIST SU (word)"),
    ("DSER_word", "DSER (word)")]
)

incremental_result_to_latex_dict = OrderedDict((key, val) for key, val in [
    ("t_t_detection_<rms_word", """TTD$_{rms}$ (word)"""),
    ("t_t_detection_<rps_word", """TTD$_{rps}$ (word)"""),
    ("t_t_detection_<e_word", """TTD$_{e}$ (word)"""),
    ("t_t_detection_t/>_word", """TTD$_{tto}$ (word)"""),
    ("t_t_detection_<rps_interval", """TTD$_{rps}$ (time in s)"""),
    ("t_t_detection_<e_interval", """TTD$_{e}$ (time in s)"""),
    ("t_t_detection_t/>_interval", """TTD$_{tto}$ (time in s)"""),
    ("edit_overhead_rel_word", "EO (word)"),
    ("edit_overhead_rel_interval", "EO (interval)")]
)


def convert_to_latex(results, eval_level=["word", "interval"],
                     inc=False, utt_seg=False,
                     only_include=None):
    """Returns a latex style tabular from results dict.
    Also displays the pandas data frame.
    """
    if not inc:
        result_to_latex_dict = final_result_to_latex_dict
    else:
        result_to_latex_dict = incremental_result_to_latex_dict
    system_results = {sys: [] for sys in results.keys()}
    utt_seg_measures = FINAL_OUTPUT_TTO_ACCURACY_HEADER.split(',') + \
        INCREMENTAL_OUTPUT_TTO_ACCURACY_HEADER.split(',')
    raw_header = []
    for raw in ACCURACY_HEADER.split(","):
        if not utt_seg and raw in utt_seg_measures:
            # print "skipping 1", raw
            continue
        for e in eval_level:
            raw = raw.format(e)
            if raw not in result_to_latex_dict.keys():
                # print "skipping 2", raw
                continue
            raw_header.append(raw)
    if only_include:
        raw_header = only_include
    # print raw_header
    header = []
    for h in raw_header:
        # print h, "*"
        conversion = result_to_latex_dict[h]
        header.append(conversion)
        for sys in results.keys():
            if "asr" in sys and "_word" in h:
                result = "-"
            else:
                result = results[sys][h]
                three_deci = ["f1_<", "t_t_d", 'correl', "r_<", "p_<"]
                if any([x in h for x in three_deci]):
                    result = '{0:.3f}'.format(result)
                else:
                    result = '{0:.3f}'.format(result)
            system_results[sys].append(result)
    rows = []
    for sys in sorted(system_results.keys()):
        corpus = "transcript"
        if 'asr' in sys:
            corpus = "ASR results"
        system = sys.split("_")[-1]
        row = [system + " ({0})".format(corpus)]
        for r in system_results[sys]:
            row.append(r)
        row = tuple(row)
        rows.append(row)
    table = pd.DataFrame(rows, columns=['System (eval. method)'] + header)
    return table


def extract_accuracies_from_file(accuracyFile, learningCurveDict):
    accuracyFile = open(accuracyFile)
    e = 1
    for line in accuracyFile:
        data = line.split()
        if len(data) > 0 and ">" in data[0]:
            e += 1
            # continue #get the even one
            # print e/2
            # print data
            # get the first one
            if e % 2 != 0:
                tag = data[0]
                # print tag
                if not learningCurveDict.get(tag) == None:
                    learningCurveDict[tag].append(float(data[3]))
            # now the rest
            for line in accuracyFile:
                data = line.split()
                if len(data) < 1:
                    break
                if e % 2 == 0:
                    tag = data[0]
                    if not learningCurveDict.get(tag) == None:
                        learningCurveDict[tag].append(float(data[3]))

                # if e > limit: break
    accuracyFile.close()


def my_legend(axis=None):

    if axis is None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    print Nlines

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((Nlines, N, N), dtype=np.float)

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin, ymin]) / ([xmax - xmin, ymax - ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot
        mask = (xy[:, 0] >= 0) & (xy[:, 0] < N) & (xy[:, 1] >= 0) & \
            (xy[:, 1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0
    # don't use the borders
    ws[:, 0] = 0
    ws[:, N - 1] = 0
    ws[0, :] = 0
    ws[N - 1, :] = 0

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N / 5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(Nlines, dtype=np.float)
        w[l] = 0.5

        # calculate a field
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
    #    plt.figure()
    #    plt.imshow(p, interpolation='nearest')
    #    plt.title(axis.lines[l].get_label())#

        pos = np.argmax(p)  # note, argmax flattens the array first
        best_x, best_y = (pos / N, pos % N)
        x = xmin + (xmax - xmin) * best_x / N
        y = ymin + (ymax - ymin) * best_y / N

        if "e" in axis.lines[l].get_label():
            y = 0.22
        if "rm-" not in axis.lines[l].get_label():
            x = 7
        else:
            x = 12


def accuracyCurvePlot(my_accuracies, limit, filename, upperlimit=None):
    """Plots the learning curve against the number of epochs.
    """
    epoch_numbers = []
    print "plotting learning curve"
    plt.gca().set_color_cycle(['red', 'green', 'blue'])
    plt.clf()
    legendlist = []
    for i in xrange(1, limit + 1):
        epoch_numbers.append(i)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 1, 1)
    plt.xlim((1, limit))
    plt.ylim((0.0, upperlimit))
    for key, val in sorted(my_accuracies.items()):
        key = key.replace("><", "_").replace("<", "")\
            .replace(">", "").replace("/", "")
        key = key.replace("rpEndSub", "rpSub")
        key = key.replace("_rpMid", "")
        plt.plot(epoch_numbers, val, label=key)
        legendlist.append(key)
    plt.legend(bbox_to_anchor=(1.02, -0.05, 0.25, 1.0), loc='top right',
               mode="expand", borderaxespad=0.)
    plt.xlabel('epochs', size='large')
    plt.ylabel('F1 accuracy for tag', size='large')
    my_legend()
    plt.savefig(filename, format="pdf")
    plt.show()
    return
