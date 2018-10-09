from mumodo.mumodoIO import open_intervalframe_from_textgrid
import numpy
from deep_disfluency.utils.accuracy import wer


final_file = open('wer_test.text', "w")
ranges1 = [line.strip() for line in open(
    "/media/data/jh/simple_rnn_disf/rnn_disf_detection/data/disfluency_detection/swda_divisions_disfluency_detection/SWDisfHeldoutASR_ranges.text")]
ranges2 = [line.strip() for line in open(
    "/media/data/jh/simple_rnn_disf/rnn_disf_detection/data/disfluency_detection/swda_divisions_disfluency_detection/SWDisfTestASR_ranges.text")]

for ranges in [ranges1, ranges2]:
    final_file.write("\n\n")
    for r in ranges:
        for s in ["A", "B"]:
            iframe = open_intervalframe_from_textgrid("{0}{1}.TextGrid"
                                                      .format(r, s))
            hyp = " ".join(iframe['Hyp']['text'])
            ref = " ".join(iframe['Ref']['text'])
            wer = wer(ref, hyp)
            cost = wer(ref, hyp, macro=True)
            print r, s, wer
            print>>final_file, r, s, wer, cost
final_file.close()


# Based on the results, output the 'good' ASR results
results = open("wer_test.text")

no_ho = 0
no_test = 0
ingood = True
file = open("../../../simple_rnn_disf/rnn_disf_detection/data/disfluency_detection/swda_divisions_disfluency_detection/SWDisfHeldoutASRgood_ranges.text", "w")
for l in results:
    # print l
    if l == "\n":
        print no_ho
        no_ho = 0
        file.close()
        file = open(
            "../../../simple_rnn_disf/rnn_disf_detection/data/disfluency_detection/swda_divisions_disfluency_detection/SWDisfTestASRgood_ranges.text",
            "w")
        continue
    if float(l.strip('\n').split(" ")[
             2]) < 0.4:  # both speakers are under 40% error rate- likely half decent separation
        # print l
        if ingood and "B" in l.strip("\n").split(" ")[1]:
            no_ho += 1
            #file.write(l.strip('\n').split(" ")[0]+l.strip('\n').split(" ")[1]+"\n")
            file.write(l.strip('\n').split(" ")[0] + "\n")
        ingood = True
    else:
        ingood = False
print no_ho

results.close()
file.close()
