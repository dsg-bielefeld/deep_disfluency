# script to take in asr results in JSON format to increco style (incremental asr results) format
import json
import os
import shutil
import numpy as np
from copy import deepcopy
from feature_utils import load_data_from_disfluency_corpus_file
from feature_utils import sort_into_dialogue_speakers, wer, get_diff_and_new_prefix

asr_dir = "/home/dsg-labuser/git/simple_rnn_disf/rnn_disf_detection/data/asr_results/"
div_dir = "/home/dsg-labuser/git/simple_rnn_disf/rnn_disf_detection/data/disfluency_detection/swda_divisions_disfluency_detection/"
heldout_ranges = [line.strip("\n") for line in open(div_dir + "SWDisfHeldout_ranges.text")]
test_ranges = [line.strip("\n") for line in open(div_dir + "SWDisfTest_ranges.text")]

heldout_file = open(asr_dir + "SWDisfHeldout_increco.text", "w")
test_file = open(asr_dir + "SWDisfTest_increco.text", "w")
leftout_ranges = [line.strip("\n") for line in open(asr_dir + "leftout_asr")]


# split the big disfluency marked -up files into individual file tuples
# it is possible to do the matching on the utterance level as they should have consistent mark-up between the two
disf_dir = "../data/disfluency_detection/switchboard"
disfluency_files = [
    disf_dir + "/swbd_heldout_partial_data.csv",
    disf_dir + "/swbd_test_partial_data.csv"]
dialogue_speakers = []
for key, disf_file in zip(["heldout", "test"], disfluency_files):
    IDs, mappings, utts, pos_tags, labels = load_data_from_disfluency_corpus_file(disf_file)
    dialogue_speakers.extend(sort_into_dialogue_speakers(IDs, mappings, utts, pos_tags, labels))
word_pos_data = {}  # map from the file name to the data
for data in dialogue_speakers:
    dialogue, a, b, c, d = data
    word_pos_data[dialogue] = (a, b, c, d)

for key in sorted(word_pos_data.keys()):
    print key
    print word_pos_data[key][1][:100]
    break
# quit()

average_wer = []
leftout = []
pair = []  # pair of files
for filename in sorted(os.listdir(asr_dir)):
    if not ".json.txt" in filename:
        continue
    conv_no = filename.replace(".json.txt", "")[3:]
    print conv_no
    resultsfile = open(asr_dir + "/" + filename)
    json_string = "[" + "".join([line for line in resultsfile]) + "]"
    # print json_string
    resultsdict = json.loads(json_string.replace("}{", "},{"))
    resultsfile.close()
    if conv_no.replace("_r", "").replace("_l", "") in leftout_ranges:
        print "leaving out file"
        continue
    if conv_no.replace("_r", "").replace("_l", "") in heldout_ranges:
        file = heldout_file
    elif conv_no.replace("_r", "").replace("_l", "") in test_ranges:
        file = test_file
    else:
        print "not in either range"
        continue

    current = []
    count = 0
    results_string = ""
    for x in resultsdict:
        # print x.keys()
        # print x["result_index"]
        # print len(x["results"][0]['alternatives'])
        results_incremental = x["results"][0]['alternatives'][0]['timestamps']
        results = []
        for a, b, c in results_incremental:
            results.append((a, b, c))
        # print results
        if results == []:
            continue
        if current == []:
            current = results
            diff = results
        else:
            current, diff, _ = get_diff_and_new_prefix(current, results)
        # print "***", count
        # print "current", current
        # print "diff", diff
        count += 1
        time_stamp = results[-1][-1]
        if diff == []:
            continue
        results_string += "Time: " + str(time_stamp) + "\n"
        results_string += "\n".join(["\t".join([str(mya) for mya in abc]) for abc in diff])
        results_string += "\n\n"
        # raw_input()
    # break
    pair.append({conv_no: [results_string, deepcopy(current)]})
    if len(pair) == 2:
        # find out whether right or left corresponds to A and B through word error rate scores
        # try matching the first 100 words of each file
        # could also do mean deviation from timings
        assigned = ""
        for single in pair:
            p = single.keys()[0]
            print p
            conv_no_overall = p.replace("_r", "").replace("_l", "")
            bestWER = 1000
            best = ""
            if assigned == "":
                for speaker in ["A", "B"]:
                    first100 = [x.replace("$unc", "").replace("$", "").lower() for x in word_pos_data[conv_no_overall + speaker][1]]

                    f100 = [x[0].replace("%HESITATION", "uh") for x in single[p][1]]
                    first100results = []
                    for word in f100:
                        capital = True
                        for letter in word:
                            if not letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                                capital = False
                        if capital and len(word) > 1:
                            first100results.extend([x.lower().encode('utf8') for x in word])
                        else:
                            first100results.append(word.lower().replace("'", "").encode('utf8'))
                        # print first100results
                        # raw_input()

                    print len(first100)
                    print len(first100results)
                    WER = wer(" ".join(first100).split(), " ".join(first100results).split())
                    print first100
                    print first100results
                    print "wer", WER
                    if WER < bestWER:
                        bestWER = WER
                        best = conv_no_overall + speaker
                        assigned = speaker
                single[p].extend([bestWER, best])

            else:
                # TODO avoiding doing another WER here
                if assigned == "A":
                    single[p].extend([0, conv_no_overall + "B"])
                else:
                    single[p].extend([0, conv_no_overall + "A"])

        print pair[0][pair[0].keys()[0]][-1]
        print pair[1][pair[1].keys()[0]][-1]
        if pair[0][pair[0].keys()[0]][-1] == pair[1][pair[1].keys()[0]][-1]:
            print "no winner!!!"
            raw_input()
            leftout.append(conv_no.replace("_r", "").replace("_l", ""))
        else:
            for single in pair:
                filename = single[single.keys()[0]][-1]
                results = single[single.keys()[0]][0].replace("%HESITATION", "uh")
                average_wer.append(single[single.keys()[0]][-2])
                file.write("File: " + filename + "\n")
                file.write(results)
        pair = []
        # break
heldout_file.close()
test_file.close()
leftoutfile = open("leftout", "w")
for l in leftout:
    print>>leftoutfile, l
leftoutfile.close()
print np.average(average_wer), "av WER"
