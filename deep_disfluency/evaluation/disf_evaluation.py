"""
Evaluation script for disfluency detection and utterance segmentation.
Takes as input a ground truth file of tab separated format:
interval_ID, start_time, end_time, word, POS_tag, ground_truth_tag.
E.g.

Speaker: KB3_1

KB3_1:1    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
KB3_1:2    1.12    2.00     $because    IN    <rps id="1"/><cc/>
KB3_1:3    2.00    3.00    because    IN    <f/><cc/>
KB3_1:4    3.00    4.00    theres    EXVBZ    <f/><cc/>
KB3_1:6    4.00    5.00    a    DT    <f/><cc/>
KB3_1:7    6.00    7.10    pause    NN    <f/><cc/>

or

Speaker: 4519A

(1:0:1)    2.557375    2.722875    your    PRP$    <f/><tc/>
(1:0:2)    2.722875    3.011125    turn    NN    <f/><ct/>
(3:7:1)    25.596    26.178875    well    UH    <e/><tc/>
(3:7:3)    26.178875    26.59325    i    PRP    <f/><cc/>
(3:7:4)    27.256125    27.460125    have    VBP    <f/><cc/>
(3:7:5)    27.50525    27.591625    to    TO    <f/><cc/>
(3:7:6)    27.591625    27.771625    say    VB    <f/><cc/>
(3:7:7)    27.771625    27.941625    i    PRP    <f/><cc/>
(3:7:8)    27.941625    28.201625    really    RB    <f/><cc/>
(3:7:9)    28.201625    28.431625    dont    VBPRB    <f/><cc/>

Second input either a file in a similar format with the final output
from the system being evaluated, modulo the reference column, e.g.:

Speaker: 4519A

2.557375    2.722875    your    PRP$    <f/><tc/>
2.722875    3.011125    turn    NN    <f/><ct/>
25.596    26.178875    well    UH    <e/><tc/>
26.178875    26.59325    i    PRP    <f/><cc/>
27.256125    27.460125    have    VBP    <f/><cc/>


Or, if you want incremental evaluation as well,
provide the increco-style tagger output for
each dialogue's speaker.
The increco-style outputs are in the following format for each output
time:

Time: time_of_update
interval_ID, start_time, end_time, word_hyp, POS_tag_hyp, predicted_tag

E.g.
Speaker: KB3_1

Time: 1.50
KB3_1:1    0.00    1.12    $unc$yes    NNP    <f/><tc/>

Time: 2.10
KB3_1:1    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
KB3_1:2    1.12    2.00     because    IN    <rps id="1"/><cc/>

Time: 2.5
KB3_1:2    1.12    2.00     because    IN
<rps id="1"/><rpndel id="1"/><cc/>

Time: 4.3
KB3_1:4    3.00    4.00    theres    EXVBZ    <f/><cc/>

Time: 5.1
KB3_1:6    4.00    5.00    a    DT    <f/><cc/>

Time: 7.25
KB3_1:7    6.00    7.10    pause    NN    <f/><cc/>

Time: 7.30
KB3_1:7    6.00    7.10    pause    NN    <f/><ct/>

The full evaluation method returns the non-incremental (final) and
incremental  disfluency evaluation as in Hough & Purver 2014 and
Hough & Schlangen 2015.
Also returns the eval on utterance semgmentation for Hough & Schlangen
2017 EACL
.
Both these evals can be done on the word-level (assuming transcripts)
or, in terms of time, as described in Hough & Schlangen 2017 EACL.

There is also a speaker rate output and error analysis output.

Note, for now, if you don't have word timings in your hypothesis
or ground truth files, there are methods in
eval_utils.py to transform the file first into a fake format.
for both the incremental and non-incremental versions.
"""
from __future__ import division

import os.path
from copy import deepcopy
from scipy.stats.stats import pearsonr, spearmanr
import numpy as np

from eval_utils import get_tag_data_from_corpus_file
from eval_utils import load_incremental_outputs_from_increco_file
from eval_utils import load_final_output_from_file
from eval_utils import p_r_f, NIST_SU, DSER
from eval_utils import final_output_accuracy_interval_level
from eval_utils import final_output_accuracy_word_level
from eval_utils import final_hyp_from_increco_and_incremental_metrics
from eval_utils import rename_all_repairs_in_line_with_index

# individual tags of interest
ACC_TAGS = ["<rms", "<rm", "<i", "<e", "<rps", "<rp", "<rpn",
            "<rpnrep", "<rpnsub", "<rpndel", "t/>"]
# combined tags of interest
COMBINED_ACC_TAGS = ["<rm.<i.<rp"]
# time to detection tags of interest
TTD_TAGS = ["<rms", "<rps", "<e", "t/>"]
RELAXED_TAGS = ["<rps", "<e", "t/>"]
ERROR_ANALYSIS_TAGS = ["<rps", "<e", "t/>", "<rms"]

MODEL_INFO_HEADER = "eval_corpus,model,context_win,\
ed_num,rp_num,rm_num,rpn_num,\
training_corpus"

# For headers, {0} argument is either word or interval
FINAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER = "p_<rm_{0},r_<rm_{0},\
f1_<rm_{0},\
p_<rm.<rp.<i_{0},r_<rm.<rp.<i_{0},f1_<rm.<rp.<i_{0},\
p_<rps_{0},r_<rps_{0},f1_<rps_{0},\
p_<rps_relaxed_{0},r_<rps_relaxed_{0},f1_<rps_relaxed_{0},\
p_<e_{0},r_<e_{0},f1_<e_{0},\
p_<e_relaxed_{0},r_<e_relaxed_{0},f1_<e_relaxed_{0}"

FINAL_OUTPUT_DISFLUENCY_RATE_ACCURACY_HEADER = "pearson_r_correl_rps_number,\
pearson_r_p_value_rps_number,\
pearson_r_correl_rps_rate_per_word,\
pearson_r_p_value_rps_rate_per_word,\
pearson_r_correl_rps_rate_per_utt,\
pearson_r_p_value_rps_rate_per_utt,\
spearman_rank_correl_rps_number,\
spearman_rank_p_value_rps_number,\
spearman_rank_correl_rps_rate_per_word,\
spearman_rank_p_value_rps_rate_per_word,\
spearman_rank_correl_rps_rate_per_utt,\
spearman_rank_p_value_rps_rate_per_utt"

INCREMENTAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER = "delayed_acc_<rm_1_{0},\
delayed_acc_<rm_2_{0},delayed_acc_<rm_3_{0},\
delayed_acc_<rm_4_{0},\
delayed_acc_<rm_5_{0},delayed_acc_<rm_6_{0},\
delayed_acc_<rm_mean_{0},\
t_t_detection_<rms_{0},t_t_detection_<rps_{0},\
t_t_detection_<e_{0},\
processing_overhead_{0},edit_overhead_rel_<rm"

FINAL_OUTPUT_TTO_ACCURACY_HEADER = "p_t/>_{0},r_t/>_{0},f1_t/>_{0},\
p_t/>_{0},\ r_t/>_relaxed_{0},f1_t/>_relaxed_{0},NIST_SU,DSER,SegER"

INCREMENTAL_OUTPUT_TTO_ACCURACY_HEADER = "t_t_detection_t/>_{0},\
t_t_detection_final_t/>_{0},\
edit_overhead_rel_tto"

ACCURACY_HEADER = ",".join([FINAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER,
                            FINAL_OUTPUT_TTO_ACCURACY_HEADER,
                            INCREMENTAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER,
                            INCREMENTAL_OUTPUT_TTO_ACCURACY_HEADER,
                            "edit_overhead_rel",
                            FINAL_OUTPUT_DISFLUENCY_RATE_ACCURACY_HEADER])

# Separate file for speaker rate
SPEAKER_RATE_HEADER = "corpus,conversation_no,speaker,\
total_turns,total_words,\
rps_hyp,rps_gold,\
rps_rate_per_utt_hyp,rps_rate_per_utt_gold,\
rps_rate_words_hyp,rps_rate_words_gold"

# Separate file for error analysis
ERROR_ANALYSIS_HEADER = "p_rms,r_rms,f1_rms,\
p_rps,r_rps,f1_rps,\
p_rpn,r_rpn,f1_rpn,\
p_rpn_rep,r_rpn_rep,f1_rpn_rep,\
p_rpn_sub,r_rpn_sub,f1_rpn_sub,\
p_rpn_del,r_rpn_del,f1_rpn_del,\
p_rps_rep,r_rps_rep,f1_rps_rep,\
p_rps_sub,r_rps_sub,f1_rps_sub,\
p_rps_del,r_rps_del,f1_rps_del,\
p_rps_rep_in_training,r_rps_rep_in_training,\
f1_rps_rep_in_training,\
p_rps_sub_in_training,r_rps_sub_in_training,\
f1_rps_sub_in_training,\
p_rps_del_in_training,r_rps_del_in_training,\
f1_rps_del_in_training,\
p_rps_rep_novel,r_rps_rep_novel,f1_rps_rep_novel,\
p_rps_sub_novel,r_rps_sub_novel,f1_rps_sub_novel,\
p_rps_del_novel,r_rps_del_novel,f1_rps_del_novel,\
"


def div(enum, denom):
    if denom == 0.0 or enum == 0.0:
        return 0.0
    return enum / denom


def final_output_disfluency_eval(prediction_speakers_dict,
                                 gold_speakers_dict,
                                 utt_eval=False,
                                 error_analysis=False,
                                 word=True,
                                 interval=True,
                                 results=None,
                                 outputfilename=None):
    """
    Non-incremental (dialogue-final) eval results.
    Returns a dict with all the required results as shown in the
    appropriate headers. Gets predictions in dicts from speakerID
    to tuples of (start, end, word, pos, goldtag), e.g.:

    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
    1.12    2.00     $because    IN    <rps id="1"/><cc/>
    2.00    3.00    because    IN    <f/><cc/>
    3.00    4.00    theres    EXVBZ    <f/><cc/>
    4.00    5.00    a    DT    <f/><cc/>
    6.00    7.10    pause    NN    <f/><cc/>

    Keyword arguments:
     prediction_speakers_dict -- dictionary from speakerID key to
         value of
         (time span,words,predictions)
     gold_speakers_dict -- dictionary from speakerID key to
         value of
         (time span, words, gold labels)
     utt_eval -- boolean, whether doing end/beginning of utterance
         evaluation too, or not
     error_analysis -- boolean, whether to do incremental error analysis
     word -- boolean, whether evaluating at the word level
     interval -- boolean, whether evaluating on the level of intervals
     results -- dict with other results, default None
     outputfilename -- path to file where final outputs are saved as
     text
    """
    print "final output disfluency evaluation"
    print "word=", word, "interval=", interval, "utt_eval=", utt_eval
    ttd_tags = deepcopy(TTD_TAGS)
    acc_tags = deepcopy(ACC_TAGS)
    relaxed_tags = deepcopy(RELAXED_TAGS)
    combined_acc_tags = deepcopy(COMBINED_ACC_TAGS)
    error_analysis_tags = deepcopy(ERROR_ANALYSIS_TAGS)
    if not utt_eval:
        if "t/>" in ttd_tags:
            ttd_tags.remove("t/>")
        if "t/>" in acc_tags:
            acc_tags.remove("t/>")
        if "t/>" in relaxed_tags:
            relaxed_tags.remove("t/>")
    if not results:
        results = {}

    if word:
        results.update({key: None for key in
                        FINAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER
                        .format("word").split(",")})
        if utt_eval:
            results.update({key: None for key in
                            FINAL_OUTPUT_TTO_ACCURACY_HEADER
                            .format("word").split(",")})
    if interval:
        results.update({key: None for key in
                        FINAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER
                        .format("interval").split(",")})
        if utt_eval:
            results.update({key: None for key in
                            FINAL_OUTPUT_TTO_ACCURACY_HEADER
                            .format("interval").split(",")})

    # get the tag dicts to do the counting
    tag_dict = {key: [0, 0, 0] for key in acc_tags}
    tag_dict.update({key: [0, 0, 0] for key in combined_acc_tags})
    tag_dict.update({"{0}_relaxed".format(key): [0, 0, 0] for key in
                     relaxed_tags})
    if utt_eval:
        tag_dict.update(
            {"NIST_SU": [0, 0, 0], "DSER": [0, 0], "SegER": [0, 0]})
    tag_dict_interval = deepcopy(tag_dict)
    # for every speaker populate speaker_rate_dict with values of tuples:
    # (number of repairs hyp, number of repairs gold,
    # no_turns, no_turns(?), no_utterances(?),no_words)
    speaker_rate_dict = {}
    if error_analysis:
        # error_analysis simply collects all the repairs with t
        # heir words and a small context window
        error_analysis = {}
        for tag in error_analysis_tags:
            error_analysis[tag] = {
                "TP": [],
                "FP": [],
                "FN": []
            }

    # loop through speakers
    # print output
    if outputfilename:
        outputfile = open(outputfilename, "w")
    # testfile = open("../test.text", "w")
    for s in sorted(prediction_speakers_dict.keys()):

        # print s
        if gold_speakers_dict.get(s) is None:
            s_test = s.replace("-", "")
            if gold_speakers_dict.get(s_test) is None:
                print s, "not in gold"
                continue
            else:
                gold = gold_speakers_dict[s_test]
        else:
            gold = gold = gold_speakers_dict[s]

        if outputfilename:
            outputfile.write("Speaker: " + s + "\n")
        # testfile.write("\nSpeaker: " + s + "\n")
        hyp = prediction_speakers_dict[s]

        hypwords = hyp[1]
        goldwords = gold[1]
        if word:  # assumes number of words == no of intervals
            prediction_tags = rename_all_repairs_in_line_with_index(
                [x[0] for x in hyp[2]],
                simple=not any(["<rm" in x[0] for x in hyp[2]]))
            gold_tags = list(gold[3])
            # for w, g, p in zip(goldwords, gold_tags, prediction_tags):
            #     testfile.write("\t".join([w, g, p]) + "\n")
            repairs_hyp,\
                repairs_gold,\
                number_of_utts_hyp,\
                number_of_utts_gold = \
                final_output_accuracy_word_level(goldwords,
                                                 prediction_tags,
                                                 gold_tags,
                                                 tag_dict=tag_dict,
                                                 utt_eval=utt_eval,
                                                 error_analysis=error_analysis)
            if outputfilename:
                final_words = []
                for g_word, h_word in zip(goldwords, hypwords):
                    joint_w = g_word
                    if not h_word[0] == g_word:
                        joint_w = joint_w + "@" + h_word[0]
                    final_words.append((h_word[1], h_word[2], joint_w))
                for w, g_tag, h_tag in zip(final_words,
                                           gold[3],
                                           [x[0] for x in hyp[2]]):
                    outputfile.write("\t".join([str(w[0]),
                                                str(w[1]),
                                                str(w[2]),
                                                "{0}@{1}"
                                                .format(g_tag, h_tag)]) + "\n")
                outputfile.write("\n")
        if interval:
            repairs_hyp,\
                repairs_gold,\
                number_of_utts_hyp,\
                number_of_utts_gold = \
                final_output_accuracy_interval_level(
                    hyp,
                    gold,
                    tag_dict=tag_dict_interval,
                    utt_eval=utt_eval,
                    error_analysis=error_analysis)
        if not speaker_rate_dict.get(s):
            speaker_rate_dict[s] = [0, 0, 0, 0, 0, 0]
        # oesn't matter if word based or interval based, the speaker rate
        # is based on turns
        speaker_rate_dict[s][0] += repairs_hyp
        speaker_rate_dict[s][1] += repairs_gold
        speaker_rate_dict[s][2] += number_of_utts_hyp
        speaker_rate_dict[s][3] += number_of_utts_gold
        speaker_rate_dict[s][4] += len(hypwords)
        speaker_rate_dict[s][5] += len(goldwords)
    if outputfilename:
        outputfile.close()
    # the accuracy calculations
    for eval_mode in ["word", "interval"]:
        if (not word) and eval_mode == "word":
            continue
        if (not interval) and eval_mode == "interval":
            continue
        if eval_mode == "interval":
            this_tag_dict = tag_dict_interval
        else:
            this_tag_dict = tag_dict
        print eval_mode
        for tag in acc_tags:
            p, r, f1 = p_r_f(this_tag_dict[tag][0],
                             this_tag_dict[tag][1],
                             this_tag_dict[tag][2])
            results["p_{0}_{1}".format(tag, eval_mode)] = p
            results["r_{0}_{1}".format(tag, eval_mode)] = r
            results["f1_{0}_{1}".format(tag, eval_mode)] = f1
        for tag in combined_acc_tags:
            TPS = 0
            FPS = 0
            FNS = 0
            for subtag in tag.split("."):
                TPS += this_tag_dict[subtag][0]
                FPS += this_tag_dict[subtag][1]
                FNS += this_tag_dict[subtag][2]
            p, r, f1 = p_r_f(TPS, FPS, FNS)
            results["p_{0}_{1}".format(tag, eval_mode)] = p
            results["r_{0}_{1}".format(tag, eval_mode)] = r
            results["f1_{0}_{1}".format(tag, eval_mode)] = f1
        if utt_eval:
            # NB sanity check the way this is done words vs. intervals,
            # should be exactly the same in transcript case
            results["NIST_SU_{0}".format(eval_mode)] = \
                NIST_SU(this_tag_dict["NIST_SU"])
            # results["SegER_{0}".format(eval_mode)] = \
            # SegER(this_tag_dict["SegER"])
            results["DSER_{0}".format(eval_mode)] = \
                DSER(this_tag_dict["DSER"])
        # relaxed: either accuracy is per turn (word) or per window (time)
        for tag in relaxed_tags:
            results["p_{0}_relaxed_{1}".format(tag, eval_mode)], \
                results["r_{0}_relaxed_{1}".format(tag, eval_mode)], \
                results["f1_{0}_relaxed_{1}".format(tag, eval_mode)] = \
                p_r_f(this_tag_dict["{0}_relaxed".format(tag)][0],
                      this_tag_dict["{0}_relaxed".format(tag)][1],
                      this_tag_dict["{0}_relaxed".format(tag)][2])

    # Speaker rate info, can be output for info
    # Now extend with 4 final columns for
    # hyp rate turn, gold rate turn, hyp rate word, gold rate word
    hyp_number_all = []
    gold_number_all = []
    hyp_rate_turn_all = []
    gold_rate_turn_all = []
    hyp_rate_word_all = []
    gold_rate_word_all = []

    for key, val in speaker_rate_dict.items():
        hyp_rate_turn = 0 if val[0] == 0 or val[2] == 0 else val[0] / val[2]
        hyp_rate_word = 0 if val[0] == 0 or val[4] == 0 else val[0] / val[4]
        gold_rate_turn = 0 if val[1] == 0 or val[5] == 0 else val[1] / val[3]
        gold_rate_word = 0 if val[1] == 0 or val[5] == 0 else val[1] / val[5]
        # add them to the dict if we want to check them
        speaker_rate_dict[key].extend([hyp_rate_turn,
                                       gold_rate_turn,
                                       hyp_rate_word,
                                       gold_rate_word])
        hyp_number_all.append(val[0])
        gold_number_all.append(val[1])
        hyp_rate_turn_all.append(hyp_rate_turn)
        hyp_rate_word_all.append(hyp_rate_word)
        gold_rate_turn_all.append(gold_rate_turn)
        gold_rate_word_all.append(gold_rate_word)

    # Now we can get the relevant Pearson correlations
    results["pearson_r_correl_rps_number"], \
        results["pearson_r_p_value_rps_number"] = \
        pearsonr(hyp_number_all, gold_number_all)
    results["pearson_r_correl_rps_rate_per_word"], \
        results["pearson_r_p_value_rps_rate_per_word"] = \
        pearsonr(hyp_rate_word_all, gold_rate_word_all)
    results["pearson_r_correl_rps_rate_per_utt"], \
        results["pearson_r_p_value_rps_rate_per_utt"] = \
        pearsonr(hyp_rate_turn_all, gold_rate_turn_all)
    results["spearman_rank_correl_rps_number"], \
        results["spearman_rank_p_value_rps_number"] = \
        spearmanr(hyp_number_all, gold_number_all)
    results["spearman_rank_correl_rps_rate_per_word"], \
        results["spearman_rank_p_value_rps_rate_per_word"] = \
        spearmanr(hyp_rate_word_all, gold_rate_word_all)
    results["spearman_rank_correl_rps_rate_per_utt"], \
        results["spearman_rank_p_value_rps_rate_per_utt"] = \
        spearmanr(hyp_rate_turn_all, gold_rate_turn_all)
    # return the results, speaker disfluency rates per speaker
    # and error analysis
    # testfile.close()
    return results, speaker_rate_dict, error_analysis


def final_output_disfluency_eval_from_file(prediction_filename,
                                           gold_speakers_dict,
                                           utt_eval=False,
                                           error_analysis=False,
                                           word=True,
                                           interval=False,
                                           results=None,
                                           outputfilename=None):
    final_output = load_final_output_from_file(prediction_filename)
    return final_output_disfluency_eval(final_output,
                                        gold_speakers_dict,
                                        utt_eval=utt_eval,
                                        error_analysis=error_analysis,
                                        word=word, interval=interval,
                                        results=results,
                                        outputfilename=outputfilename)


def incremental_output_disfluency_eval(prediction_speakers_dict,
                                       gold_speakers_dict,
                                       utt_eval=False,
                                       error_analysis=False,
                                       word=True,
                                       interval=False,
                                       outputfilename=None):
    """
     The incremental results from an increco style outputs from the
     system and
     the gold standard (adjusted for repair tags).
     Also gives the final/stable output results for each sequence
     consumed.
     This will also give results on the segment level (which is
     isomorphic to
     the word level when using gold standard transcripts)

     Keyword arguments:
     prediction_speakers_dict -- dictionary from speakerID key to value:
         (start_time, end_time,words,predictions)
     gold_speakers_dict -- dictionary from speakerID key to value@:
         (start_time, end_time, words, gold labels)
     utt_eval -- boolean, whether doing utterance segmenation evaluation
     error_analysis -- boolean, whether to do incremental error analysis
     word -- boolean, whether evaluating at the word level
     interval -- boolean, whether evaluating on the level of intervals
     results -- dict with other results, default None
     outputfilename -- path to file where final outputs are saved as
     text
    """
    print "incremental output disfluency evaluation"
    print "word=", word, "interval=", interval, "utt_eval=", utt_eval
    ttd_tags = deepcopy(TTD_TAGS)
    acc_tags = deepcopy(ACC_TAGS)
    relaxed_tags = deepcopy(RELAXED_TAGS)
    if not utt_eval:
        if "t/>" in ttd_tags:
            ttd_tags.remove("t/>")
        if "t/>" in acc_tags:
            acc_tags.remove("t/>")
        if "t/>" in relaxed_tags:
            relaxed_tags.remove("t/>")
    results = {}
    tag_dict = {}
    if word:
        results.update(
            {key: None for key in
             INCREMENTAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER
             .format("word").split(",")})
        tag_dict.update(
            {"t_t_detection_{0}_{1}".format(key, "word"): []
             for key in ttd_tags})
        if utt_eval:
            results.update(
                {key: None for key in
                 INCREMENTAL_OUTPUT_TTO_ACCURACY_HEADER
                 .format("word").split(",")})
    if interval:
        results.update({key: None for key in
                        INCREMENTAL_OUTPUT_DISFLUENCY_ACCURACY_HEADER
                        .format("interval").split(",")})
        tag_dict.update(
            {"t_t_detection_{0}_{1}".format(key, "interval"): []
             for key in ttd_tags})
        if utt_eval:
            results.update({key: None for key in
                            INCREMENTAL_OUTPUT_TTO_ACCURACY_HEADER
                            .format("interval").split(",")})
    tag_dict.update({"edit_overhead": [0, 0]})
    # started = False
    for s in sorted(prediction_speakers_dict.keys()):
        carry_on = True  # False
        if s == "4611A":
            carry_on = True
        if not carry_on:
            continue
        if gold_speakers_dict.get(s) is None:
            print s, "not in gold"
            continue
        hyp = prediction_speakers_dict[s]
        gold = [(x, y[0], y[1])
                for x, y in zip(gold_speakers_dict[s][3],
                                gold_speakers_dict[s][0])]
        # we do word + interval in one swoop
        final_hyp = final_hyp_from_increco_and_incremental_metrics(
            hyp,
            gold,
            gold_speakers_dict[s][1],
            utt_eval,
            ttd_tags=ttd_tags,
            word=word,
            interval=interval,
            tag_dict=tag_dict,
            speaker_ID=s)
        # replace the incremental results with the final one only
        prediction_speakers_dict[s] = final_hyp
        # print gold_speakers_dict[s][1]
        #final_gold_words = [x[0] for x in gold_speakers_dict[s][1]]
        # print final_gold_words
        # gold_speakers_dict[s] = (gold_speakers_dict[s][0],
        #                         final_gold_words,
        #                         gold_speakers_dict[s][2],
        #                         gold_speakers_dict[s][3])
        # break #todo remove
    # do the final output eval too
    for eval_mode in ["word", "interval"]:
        if (not word) and eval_mode == "word":
            continue
        if (not interval) and eval_mode == "interval":
            continue

        results["edit_overhead_rel_{}".format(
            eval_mode)] = 100 * ((tag_dict["edit_overhead"][0] /
                                  tag_dict["edit_overhead"][1]) - 1)
        for t_tag in ttd_tags:
            results["t_t_detection_{0}_{1}".format(t_tag, eval_mode)] = \
                np.average(
                tag_dict["t_t_detection_{0}_{1}".format(t_tag, eval_mode)]
            )
#     results, speaker_rate_dict, error_analysis = final_output_disfluency_eval(
#         prediction_speakers_dict,
#         gold_speakers_dict,
#         utt_eval=utt_eval,
#         error_analysis=error_analysis,
#         word=word,
#         interval=interval,
#         results=results,
#         outputfilename=outputfilename)
    if outputfilename:
        print "writing final output to file", outputfilename
        outputfile = open(outputfilename, "w")
        for s in sorted(prediction_speakers_dict.keys()):

            # print s
            if gold_speakers_dict.get(s) is None:
                print s, "not in gold"
                continue
            if outputfilename:
                outputfile.write("Speaker: " + s + "\n")
            hyp = prediction_speakers_dict[s]
            gold = gold_speakers_dict[s]
            hypwords = hyp[1]
            goldwords = gold[1]
            final_words = []
            for g_word, h_word in zip(goldwords, hypwords):
                joint_w = g_word
                if not h_word[0] == g_word:
                    joint_w = joint_w + "@" + h_word[0]
                final_words.append((h_word[1], h_word[2], joint_w))
            for w, g_tag, h_tag in zip(final_words,
                                       gold[3],
                                       [x[0] for x in hyp[2]]):
                outputfile.write("\t".join([str(w[0]),
                                            str(w[1]),
                                            str(w[2]),
                                            "{0}@{1}"
                                            .format(g_tag, h_tag)]) + "\n")
                outputfile.write("\n")
            outputfile.write("\n")
        outputfile.close()
    return results, error_analysis


def incremental_output_disfluency_eval_from_file(prediction_filename,
                                                 gold_speakers_dict,
                                                 utt_eval=False,
                                                 error_analysis=False,
                                                 word=True,
                                                 interval=False,
                                                 outputfilename=None):
    final_output = load_incremental_outputs_from_increco_file(
        prediction_filename)
    return incremental_output_disfluency_eval(final_output,
                                              gold_speakers_dict,
                                              utt_eval=utt_eval,
                                              error_analysis=error_analysis,
                                              word=word,
                                              interval=interval,
                                              outputfilename=outputfilename)


def save_results_to_file(test_filename,
                         results_filename,
                         speaker_rate_filename,
                         results, speaker_rate_dict, model_info):
    """Saves the accuracy results in results dictionary to
    results_filename.
    Saves the speaker rates of disfluency for each speaker to another
     speaker_rate_filename.
    Needs values from the model_info dict for all the values in the
    MODEL_INFO_HEADER
    """
    if os.path.isfile(results_filename):  # may be adding to existing file
        results_file = open(results_filename, "a")
    else:
        results_file = open(results_filename, "w")
        results_file.write(MODEL_INFO_HEADER + "," + ACCURACY_HEADER + "\n")

    results_file.write(
        ",".join([str(model_info[x])
                  for x in MODEL_INFO_HEADER.split(",")]) + ",")
    results_file.write(
        ",".join([str(results[x]) for x in ACCURACY_HEADER.split("")]) + "\n")
    results_file.close()

    if os.path.isfile(speaker_rate_filename):
        speaker_rate_file = open(speaker_rate_filename, "a")
    else:
        speaker_rate_file = open(speaker_rate_file, "w")
        speaker_rate_file.write(SPEAKER_RATE_HEADER + "\n")
    for key, val in sorted(speaker_rate_dict.items(), key=lambda x: x[0]):
        # split into the convo + participant to see how the two in the same
        # dialogue compare
        conv_no = key.split(":")[0]
        speaker = key.split(":")[1]
        speaker_rate_file.write(",".join([str(x) for x in [test_filename,
                                                           conv_no, speaker,
                                                           val[2], val[4],
                                                           val[0], val[1],
                                                           val[5], val[6],
                                                           val[7],
                                                           val[8]]]) + "\n")
    speaker_rate_file.close()


if __name__ == '__main__':

    # doesn't apply to the rnn/lstm
    ed_num, rps_num, rms_num, rpn_num = None, None, None, None
    partial_string = "_partial"
    eval_corpus = "swda_test" + partial_string
    training_corpus = "swda_train"
    model = 'stir'
    context_win = 3

    # RNN files- need to conform to the increco standards.
    top_dir = "../../../simple_rnn_disf/rnn_disf_detection"
    hyp_dir = top_dir + "/experiments/035/"
    disf_dir = top_dir + "/data/disfluency_detection/switchboard"
    disfluency_files = [
        disf_dir + "/swbd_heldout{0}_timings_data.csv".format(partial_string),
        disf_dir + "/swbd_test{0}_timings_data.csv".format(partial_string),
    ]

    dialogue_speakers = []
    for key, disf_file in zip(["swbd_disf_heldout",
                               "swbd_disf_test"], disfluency_files):
        # if not key == "heldout": continue
        IDs, mappings, utts, pos_tags, labels = \
            get_tag_data_from_corpus_file(disf_file)
        gold_data = {}  # map from the file name to the data
        for dialogue, a, b, c, d in zip(IDs,
                                        mappings,
                                        utts,
                                        pos_tags,
                                        labels):
            gold_data[dialogue] = (a, b, c, d)

        e = 9
        results = incremental_output_disfluency_eval_from_file(
            hyp_dir + "epoch_{0}/{1}{2}_output_increco.text".format(
                e, key, partial_string),
            gold_data,
            utt_eval=True,
            error_analysis=False,
            word=True,
            interval=False,
            outputfilename=hyp_dir +
            "epoch_{0}/{1}{2}_output_final.text".format(e, key,
                                                        partial_string))
        for k, v in results.items():
            print k, v
