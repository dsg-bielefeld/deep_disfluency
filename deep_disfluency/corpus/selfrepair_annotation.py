# -*- coding: utf-8 -*-
from collections import defaultdict
from operator import itemgetter

from swda import CorpusReader
from tree_pos_map import TreeMapCorpus, POSMapCorpus
from util import clean, parse_list


class SelfRepair:

    def __init__(self):
        self.errorlog = open('errors.txt', 'w')
        self.corpus = CorpusReader('..\swda')
        self.__1plist__ = []  # within-turn self-repairs
        # transition space repairs and abandoned turns followed by restart
        # (fresh starts)
        self.__2plist__ = []
        self.__3plist__ = []  # repair with interleaved hearer material or NTRI
        self.__lists__ = []  # list of all the lists that are used

        self.extras = []  # unannotated bits

        self.forward_acts = ["sv", "qw^d", "qh", "sd", "qy", "qw",
                             "qy^d", "qrr", "qo", "ad", "oo_co_cc",
                             "fc", "fa", "fp"]
        self.backward_acts = ["+", "h", "aap_aam", "arp_nd", "no", "bf",
                              "na", "ng", "^q",
                              "ar", "^2", "ba", "br", "aa", "bd", "b^m",
                              "ft", "nn", "bh", "bk", "ny", "b"]
        # others will be other! NB have on one occasion, file 97 added a "a/oo" marker with forward backslash to show change of dialogue act
        # self.test()
        # self.show_tree(1,24,0)

        self.__treeMapList__ = TreeMapCorpus(True, self.errorlog)
        self.__posMapList__ = POSMapCorpus(True, self.errorlog)
        # self.show_tree(164,45,0)

        # add all the range files- just do SW2 simple, no tree stuff
        self.ranges = []
        rangeFile = open("ranges\\SW1Full_ranges.text", "r")
        for line in rangeFile:
            a = line.replace("\n", "")
            self.ranges.append(a)
        rangeFile.close()
        # rangeFile = open("ranges\\SWJCHeldout_ranges.text","r") #get this done too
        # for line in rangeFile:
        #    a = line.replace("\n","")
        #    self.ranges.append(a)

        print "files in ranges = " + str(len(self.ranges))

        self.filename = "SW1Full"
        #self.file1p = open("Annotations\p1RepairsNgram_"+ self.filename +".csv.text","w")
        #self.file3p = open("Annotations\p3RepairsNgram_"+ self.filename +".csv.text","w")

        # annotation stuff
        """
        markerfile = open("AnnotationMarker.text")
        for line in markerfile:
            #gives the reparandumStart of the last unannotated pair
            marker = line.split(",")
        self.unannotated_marker = []
        for number in marker:
            self.unannotated_marker.append(int(number))
        self.unannotated_marker = tuple(self.unannotated_marker)
        markerfile.close()
        markerfile = open("AnnotationMarker.text","w")
        #re-write the files:
        try:
            self.self_repairs()
        except:
            print sys.exc_info()
            print "ERROR, EXITING AND SAVING!"
            self.file1p.close()
            self.file3p.close()
            unanstring = ""
            for string in self.unannotated_marker:
                unanstring +=str(string) + ","
            unanstring= unanstring[:-1]
            markerfile.write(unanstring); markerfile.close()
        """
        # self.edit_types(file1pname,file3pname) # extend each list by one by
        # adding edit type to the end via manual annotation

        # self.self_repairs()

        # for repair in self.__1plist__:
        #    self.print_1p_NoTreeStuff(repair,self.file1p)
        # for repair in self.__3plist__:
        #    self.print_3p_NoTreeStuff(repair,self.file3p)
        # self.file1p.close()
        # self.file3p.close()

        # self.show_tree(437,63,0)
        # self.test()
        # self.data()

        self.corpus_stats()
        self.errorlog.close()

    def data(self):
        corpus = CorpusReader('..\swda')
        # Iterate through all transcripts
        trainingNumber = 0  # sw_1
        trainingUtterances = 0
        testNumber = 0  # jc test- miss these
        testUtterances = 0
        developmentNumber = 0  # sw_heldout
        developmentUtterances = 0
        semdialtestNumber = 0  # sw_test
        semdialtestUtterances = 0
        otherNumber = 0  # this should constitute SW_2
        otherUtterances = 0
        totalUtterances = 0
        missed = []  # just the JC test files

        #file = open("..\\swda\\bnc-scripts\\bnc-scripts\\SW_1_ranges.text","w")
        #file2 = open("..\\swda\\bnc-scripts\\bnc-scripts\\SW_1_SW_2_ranges.text","w")
        #file22 = open("..\\swda\\bnc-scripts\\bnc-scripts\\SW_2_ranges.text","w")
        #file3 = open("..\\swda\\bnc-scripts\\bnc-scripts\\SW_heldout_ranges.text","w")
        #file4 = open("..\\swda\\bnc-scripts\\bnc-scripts\\SW_test_ranges.text","w")
        file5 = open("..\\swda\\semdial\\ranges\\SW_JC_TEST_ranges.text", "w")
        # the only things we leave out are the JC test data

        for trans in corpus.iter_transcripts():
            transnumber = int(trans.swda_filename[19:23])
            count = 0
            if transnumber > 1210:
                break
            translength = len(trans.utterances)
            totalUtterances += len(trans.utterances)

            if int(trans.conversation_no) >= 2000 and int(
                    trans.conversation_no) < 4000:
                if trans.has_trees():  # jc train, fine for both training files
                    trainingNumber += 1
                    trainingUtterances += len(trans.utterances)
                    # file.write(trans.swda_filename+"\n")
                    # file2.write(trans.swda_filename+"\n")
                else:  # just fine for bigger training file
                    otherNumber += 1
                    otherUtterances += len(trans.utterances)
                    # file22.write(trans.swda_filename+"\n")
                    # file2.write(trans.swda_filename+"\n")
            elif int(trans.conversation_no) >= 4004 and int(trans.conversation_no) < 4154:
                if trans.has_trees():  # jc test- omit!
                    testNumber += 1
                    testUtterances += len(trans.utterances)
                    file5.write(trans.swda_filename + "\n")
                    # missed.append((trans.swda_filename,trans.conversation_no))
                    # #only things we leave out
                else:  # fine for bigger file
                    otherNumber += 1
                    otherUtterances += len(trans.utterances)
                    # file22.write(trans.swda_filename+"\n")
                    # file2.write(trans.swda_filename+"\n")
            elif int(trans.conversation_no) >= 4519 and int(trans.conversation_no) < 4937:
                if trans.has_trees():  # jc and my dev
                    developmentNumber += 1
                    developmentUtterances += len(trans.utterances)
                    # file3.write(trans.swda_filename+"\n")
                else:  # fine for bigger file
                    otherNumber += 1
                    otherUtterances += len(trans.utterances)
                    # file22.write(trans.swda_filename+"\n")
                    # file2.write(trans.swda_filename+"\n")
            else:
                # this should give us the other files for our test set
                if trans.has_trees():  # jc future, our test
                    semdialtestNumber += 1
                    semdialtestUtterances += len(trans.utterances)
                    # file4.write(trans.swda_filename+"\n")
                else:  # fine for our bigger training set
                    if trans.conversation_no == 4330:
                        raw_input()
                    otherNumber += 1
                    otherUtterances += len(trans.utterances)
                    # file22.write(trans.swda_filename+"\n")
                    # file2.write(trans.swda_filename+"\n")
        file5.close()
        print "training"
        print trainingNumber
        print trainingUtterances
        print "semdial testing"
        print semdialtestNumber
        print semdialtestUtterances
        print "dev"
        print developmentNumber
        print developmentUtterances
        print "not used (either future for JC or not in treebank)"
        print otherNumber
        print otherUtterances

        print "test for JC (the ones in PTB that that is, otherwise we can use them)"
        print testNumber
        print testUtterances
        print len(missed)
        print "sw_1_sw_2"
        print trainingNumber + otherNumber
        print trainingUtterances + otherUtterances
        raw_input()
        # for pair in missed:
        #    print pair

        print "total utts"
        print totalUtterances
        data = trainingNumber + semdialtestNumber + \
            developmentNumber + otherNumber
        print data
        print 1126 - data == testNumber
        datautts = trainingUtterances + semdialtestUtterances + \
            developmentUtterances + otherUtterances
        print datautts
        print totalUtterances - datautts == testUtterances

    def show_tree(self, filenumber, uttnumber, treenumber):
        #corpus = CorpusReader('..\swda')
        for trans in self.corpus.iter_transcripts():
            transnumber = int(trans.swda_filename[19:23])
            if transnumber == int(filenumber):
                trans.utterances[uttnumber].trees[treenumber].draw()
                mytree = trans.utterances[uttnumber].trees[
                    treenumber].pprint_latex_qtree()
                filename = "..\\swda\\qtrees\\file" + \
                    str(int(filenumber)) + "utt" + str(uttnumber) + ".text"
                file = open(filename, 'w')
                file.write(mytree)
                file.close()
                break

    def print_1p(self, repair, file):
        # input:
        # 0 transcriptFile,
        # 10(startUttNum,
        # 11 startNum),
        # 20(interregUttNum,
        # 21 interregNum),
        # 30 (repairUttNum,
        # 31 repairNum),
        # 40(endUttNum,
        # 41 endNum),
        # 5 origWords,
        # 6 reparandumWords,
        # 7 interregnumWords,
        # 8 repairWords,
        # 9 endWords
        # 10 interregTypes,
        # 11 myDepths,
        # 12 myPathlengths
        # 13 embeddedIn,
        # 14 startUtt.damsl_act_tag()
        # 15 editing types
        # what we want after 9:
        interruptionPos = len(repair[5]) + len(repair[6])
        reparandumLength = len(repair[6])
        # 10 interregTypes
        # 11 myDepths
        if len(repair[11][1]) == 0:
            self.errorlog.write("\nNo reparandum 1p!! " + repair[0] + "  " + str(
                repair[1]) + str(repair[2]) + str(repair[3]) + str(repair[4]) + "\n")
            return
        # i.e. the reparandum depths first depth
        reparandumOnsetDepth = repair[11][1][0][2]
        if len(repair[11][0]) > 0:
            # i.e the last depth in origutt before reparandum
            depthbeforeReparandum = repair[11][0][-1][2]
            reparandumOnsetChange = reparandumOnsetDepth - \
                depthbeforeReparandum
        else:
            reparandumOnsetChange = "S"
        if len(repair[11][1]) > 1:  # more than onecostit in reparandum
            change2 = repair[11][1][-1][2] - repair[11][1][-2][2]
            POS1 = repair[11][1][-2][1]
            POS2 = repair[11][1][-1][1]
            if len(repair[11][1]) > 2:
                change1 = repair[11][1][-2][2] - repair[11][1][-3][2]
            elif len(repair[11][0]) > 0:
                change1 = repair[11][1][-2][2] - repair[11][0][-1][2]
            else:
                change1 = "S"
        elif len(repair[11][0]) > 0:  # i.e. there is stuff in orig
            change2 = reparandumOnsetDepth - repair[11][0][-1][2]
            POS2 = repair[11][1][-1][1]  # last/first reparandumPOS
            POS1 = repair[11][0][-1][1]  # last/first origPOS
            if len(repair[11][0]) > 1:
                change1 = repair[11][0][-1][2] - repair[11][0][-2][2]
            else:
                change1 = "S"
        else:
            # last AND first reparandum POS- just get this one
            POS2 = repair[11][1][-1][1]
            POS1 = "<!>"
            change1 = "<!>"
            change2 = "S"
        # this is reversed! in order to get unigram type effect to easily if
        # needed
        changeBigramBeforeRepair = str(change2) + "," + str(change1)
        POSreparandumOnset = repair[11][1][0][1]
        # these are reversed to get unigram effect if needed!
        POSBigramBeforeRepair = POS2 + "," + POS1
        # 12 Pathlengths: change this
        # 13 Embedded
        # 14 DA
        # EditType   i.e. repair[15][0]
        reparandumPattern = ""
        """for string, number in repair[15][1][1]:
            if string == "f" or string == "s":
                reparandumPattern+= string + ","
            else: reparandumPattern= reparandumPattern  + string + "[" + str(number) + "]" + ","
        """
        repairPattern = ""
        """for string, number in repair[15][1][3]:
            if string == "f" or string == "s":
                repairPattern+= string + ","
            else: repairPattern= repairPattern  + string + "[" + str(number) + "]" + ","
        #editing types i.e. repair[15][1] # missing this for now/no editing type stuff
        """
        file.write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(repair[1][1]) + "\t" + str(repair[2][0]) + "\t" + str(repair[2][1]) + "\t" + str(repair[3][0]) + "\t" + str(repair[3][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t" + str(repair[5]) + "\t" + str(repair[6]) + "\t" + str(repair[7]) + "\t" + str(repair[8]) + "\t" + str(repair[9]) + "\t" + str(interruptionPos) + "\t" + str(
            reparandumLength) + "\t" + str(repair[10]) + "\t" + str(repair[11]) + "\t" + str(reparandumOnsetDepth) + "\t" + str(reparandumOnsetChange) + "\t" + str(changeBigramBeforeRepair) + "\t" + str(POSreparandumOnset) + "\t" + str(POSBigramBeforeRepair) + "\t" + str(repair[12]) + "\t" + str(repair[13]) + "\t" + str(repair[14]) + "\t" + str(reparandumPattern) + "\t" + str(repairPattern) + "\n")

    def print_3p(self, repair, file):
        #file = open("3PRepairs.text.csv.text", "w")
        # format from repair should be: text 0, numbers 1-8, text 9-20
         # 0 transcriptFile,
        # 10(startUttNum,
        # 11 startNum),
        # 20(interregUttNum,
        # 21 interregNum),
        # 30 (repairUttNum,
        # 31 repairNum),
        # 40(endUttNum,
        # 41 endNum),
        # 5 origWords,
        # 6 reparandumWords,
        # 7 interregnumWords,
        # 8 repairWords,
        # 9 endWords
        # 10 interregTypes,
        # 11 myDepths,
        # 12 myPathLengths,
        # 13 embeddedIn
        # 14 startUtt.damsl_act_tag()
        # 15 editing types
        # 16 repairDAct
        # 17 interleavedUtts
        # 18 splitUttBool,
        # 19 completeReparandumBool
        # 20 repairFromOther
           # what we want after 9:
        interruptionPos = len(repair[5]) + len(repair[6])
        reparandumLength = len(repair[6])
        # 10 interregTypes
        # 11 myDepths
        if len(repair[11][1]) == 0:
            self.errorlog.write("\nNo reparandum 3p!! " + repair[0] + "  " + str(
                repair[1]) + str(repair[2]) + str(repair[3]) + str(repair[4]) + "\n")
            return
        # i.e. the reparandum depths first depth
        reparandumOnsetDepth = repair[11][1][0][2]
        if len(repair[11][0]) > 0:
            # i.e the last depth in origutt before reparandum
            depthbeforeReparandum = repair[11][0][-1][2]
            reparandumOnsetChange = reparandumOnsetDepth - \
                depthbeforeReparandum
        else:
            reparandumOnsetChange = "S"
        if len(repair[11][1]) > 1:  # more than onecostit in reparandum
            change2 = repair[11][1][-1][2] - repair[11][1][-2][2]
            POS1 = repair[11][1][-2][1]
            POS2 = repair[11][1][-1][1]
            if len(repair[11][1]) > 2:
                change1 = repair[11][1][-2][2] - repair[11][1][-3][2]
            elif len(repair[11][0]) > 0:
                change1 = repair[11][1][-2][2] - repair[11][0][-1][2]
            else:
                change1 = "S"
        elif len(repair[11][0]) > 0:  # i.e. there is stuff in orig
            change2 = reparandumOnsetDepth - repair[11][0][-1][2]
            POS2 = repair[11][1][-1][1]  # last/first reparandumPOS
            POS1 = repair[11][0][-1][1]  # last/first origPOS
            if len(repair[11][0]) > 1:
                change1 = repair[11][0][-1][2] - repair[11][0][-2][2]
            else:
                change1 = "S"
        else:
            # last AND first reparandum POS- just get this one
            POS2 = repair[11][1][-1][1]
            POS1 = "<!>"
            change1 = "<!>"
            change2 = "S"
        # this is reversed! in order to get unigram type effect to easily if
        # needed
        changeBigramBeforeRepair = str(change2) + "," + str(change1)
        POSreparandumOnset = repair[11][1][0][1]
        # these are reversed to get unigram effect if needed!
        POSBigramBeforeRepair = POS2 + "," + POS1
        # 12 myPathlengths
        # 13 Embedded
        # 14 DA
        # EditType   i.e. repair[15][0]
        reparandumPattern = ""
        """for string, number in repair[15][1][1]:
            if string == "f" or string == "s":
                reparandumPattern+= string + ","
            else: reparandumPattern= reparandumPattern  + string + "[" + str(number) + "]" + ","
        """
        repairPattern = ""
        """for string, number in repair[15][1][3]:
            if string == "f" or string == "s":
                repairPattern+= string + ","
            else: repairPattern= repairPattern  + string + "[" + str(number) + "]" + ","
        """
        repair.insert(15, "")  # dummy for now
        # editing types i.e. repair[15][1] #missing this for now, no editing type stuff
        # then 16 as per above

        # here turning 17 interLeavedUtts into concatenation of their texts and
        # a list of DAs
        interleavedStrings = []
        interleavedDAs = []
        for utt in repair[17]:
            interleavedStrings += utt.text_words(filter_disfluency=True)
            interleavedDAs.append(utt.damsl_act_tag())
        # now leave out 17, add those two instead
        # 18 as per above
        # 19 as per above
        # 20 as above
        file.write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(repair[1][1]) + "\t" + str(repair[2][0]) + "\t" + str(repair[2][1]) + "\t" + str(repair[3][0]) + "\t" + str(repair[3][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t" + str(repair[5]) + "\t" + str(repair[6]) + "\t" + str(repair[7]) + "\t" + str(repair[8]) + "\t" + str(repair[9]) + "\t" + str(interruptionPos) + "\t" + str(reparandumLength) + "\t" + str(repair[10]) + "\t" + str(
            repair[11]) + "\t" + str(reparandumOnsetDepth) + "\t" + str(reparandumOnsetChange) + "\t" + str(changeBigramBeforeRepair) + "\t" + str(POSreparandumOnset) + "\t" + str(POSBigramBeforeRepair) + "\t" + str(repair[12]) + "\t" + str(repair[13]) + "\t" + str(repair[14]) + "\t" + str(repair[16]) + "\t" + str(interleavedDAs) + "\t" + str(interleavedStrings) + "\t" + str(len(interleavedStrings)) + "\t" + str(repair[18]) + "\t" + str(repair[19]) + "\t" + str(repair[20]) + "\n")

    def print_1p_NoTreeStuff(self, repair, file):
        # input:
        # 0 transcriptFile,
        # 10(startUttNum,
        # 11 startNum),
        # 20(interregUttNum,
        # 21 interregNum),
        # 30 (repairUttNum,
        # 31 repairNum),
        # 40(endUttNum,
        # 41 endNum),
        # 5 origWords,
        # 6 reparandumWords,
        # 7 interregnumWords,
        # 8 repairWords,
        # 9 endWords
        # 10 interregTypes,
        # 11 embeddedIn,
        # 12 startUtt.damsl_act_tag()
        # what we want after 9:
        #interruptionPos = len(repair[5]) + len(repair[6])
        #reparandumLength = len(repair[6])
        # 10 interregTypes
        # 11 embeddedIn
        # 12 startUtt.damsl_act_tag()
        #file.write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(repair[1][1]) + "\t" + str(repair[2][0]) + "\t" + str(repair[2][1]) + "\t" + str(repair[3][0]) + "\t" + str(repair[3][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t" + str(repair[5]) + "\t" + str(repair[6]) + "\t" + str(repair[7]) + "\t" + str(repair[8]) + "\t" + str(repair[9]) + "\t" + str(interruptionPos) + "\t" + str(reparandumLength) + "\t" + str(repair[10]) + "\t" + str(repair[11]) + "\t" + str(repair[12]) + "\n")
        file.write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(repair[1][1]) + "\t" + str(repair[2][0]) + "\t" + str(repair[2][1]) +
                   "\t" + str(repair[3][0]) + "\t" + str(repair[3][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t" + str(repair[5]) + "\n")

    def print_3p_NoTreeStuff(self, repair, file):
        #file = open("3PRepairs.text.csv.text", "w")
        # format from repair should be: text 0, numbers 1-8, text 9-20
        # 0 transcriptFile,
        # 10(startUttNum,
        # 11 startNum),
        # 20(interregUttNum,
        # 21 interregNum),
        # 30 (repairUttNum,
        # 31 repairNum),
        # 40(endUttNum,
        # 41 endNum),
        # 5 origWords,
        # 6 reparandumWords,
        # 7 interregnumWords,
        # 8 repairWords,
        # 9 endWords
        # 10 interregTypes,
        # 11 embeddedIn
        # 12 startUtt.damsl_act_tag()
        # 13 repairDAct
        # 14 interleavedUtts
        # 15 splitUttBool,
        # 16 completeReparandumBool
        # 17 repairFromOther
        # what we want after 9:
        #interruptionPos = len(repair[5]) + len(repair[6])
        #reparandumLength = len(repair[6])
        # 10 interregTypes
        # 11 to 13 as above
        # here turning 14 interLeavedUtts into concatenation of their texts and a list of DAs
        #interleavedStrings =[]
        #interleavedDAs = []
        # for utt in repair[14]:
        #    interleavedStrings += utt.text_words(filter_disfluency=True)
        #    interleavedDAs.append(utt.damsl_act_tag())
        # now leave out 14, add those two instead
        # ..15 and 16 and 17 as per above
        file.write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(repair[1][1]) + "\t" + str(repair[2][0]) + "\t" + str(repair[2][1]) +
                   "\t" + str(repair[3][0]) + "\t" + str(repair[3][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t" + str(repair[5]) + "\n")
        #file.write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(repair[1][1]) + "\t" + str(repair[2][0]) + "\t" + str(repair[2][1]) + "\t" + str(repair[3][0]) + "\t" + str(repair[3][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t" + str(repair[5]) + "\t" + str(repair[6]) + "\t" + str(repair[7]) + "\t" + str(repair[8]) + "\t" + str(repair[9]) + "\t" + str(interruptionPos) + "\t" + str(reparandumLength) + "\t" + str(repair[10]) + "\t" + str(repair[11]) + "\t" + str(repair[12]) + "\t" + str(repair[13]) + str(interleavedDAs)+ "\t" + str(interleavedStrings) + "\t" + str(len(interleavedStrings)) + "\t" + str(repair[15]) + "\t" + str(repair[16]) + "\t" + str(repair[17]) + "\n")

    # ad hoc method for printing the annotation positions and path lengths for
    # each repair:

    # allows you to print the path lengths of the various different types
    def print_path_lengths(self, repairlists, date):

        NUMBERBUCKETS = ([0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0], [
                         0.6, 0], [0.7, 0], [0.8, 0], [0.9, 0], [1, 0])
        firstPNUMBERBUCKETS = ([0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0], [
                               0.6, 0], [0.7, 0], [0.8, 0], [0.9, 0], [1, 0])
        thirdPNUMBERBUCKETS = ([0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0], [
                               0.6, 0], [0.7, 0], [0.8, 0], [0.9, 0], [1, 0])
        origLengthBucketsRelative = []
        interregnumDict = defaultdict(int)
        onsetPOSdict = defaultdict(int)

        # just for sentences of up to length 16, pair of origLength and buckets
        for i in range(1, 17):
            origLengthBucketsRelative.append((i, [[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [
                                             0.5, 0], [0.6, 0], [0.7, 0], [0.8, 0], [0.9, 0], [1.0, 0]]))
        origLengthBucketsRaw = []
        for i in range(1, 17):  # just for setences up to length 16
            emptylist = [["<S>", 0]]
            for k in range(1, 33):  # might go up to long lengths,
                emptylist.append([str(k), 0])
            origLengthBucketsRaw.append((i, emptylist))
        lengthdict = defaultdict(int)
        firstPlengthdict = defaultdict(int)
        thirdPlengthdict = defaultdict(int)
        # what we need- first, print the
        filelists = []
        # just do first and 3rd position for now, can expand to 1,2(ts),3
        positionlists = [1, 3]
        for repairlist in repairlists:
            filename = "SelfRepairResults\p" + \
                str(positionlists[0]) + \
                "repairPATHLENGTHS" + date + ".csv.text"
            del positionlists[0]
            filelists.append((open(filename, "w"), repairlist))
        firstP = True
        for filepair in filelists:  # iterates over all..
            filepair[0].write(
                "FILE \t STARTUTT \t STARTWORD \t ENDUTT \t ENDWORD \t")
            # raw values
            filepair[0].write(
                "ORIGUTTLENGTH \t DISTANCEBACK \t ONSETPATHLENGTH \t ONSETCOMMONMOTHER \t NEXTCOMMONMOTHER \t LASTCONSTITPATHLENGTH \t LASTCONSTITCOMMONMOTHER \t")
            # relative values (of path length)
            filepair[0].write("RELATIVEPATHLENGTH \n")
            COUNT = 0
            for repair in filepair[1]:
                COUNT += 1
                pathlengths = []
                rank = 1  # get rank of reparandum pathlength
                relativePathLength = 0  # i.e. the pathlength normalised
                # acts as null here, must have some distance back.. no
                # abridging repairs
                myDistanceBack = 0
                # get value of onset Pathlength  1 acts as dud here
                reparandumOnsetPathlength = 1
                # get value of the last constit path length
                lastConstitPathlength = 1
                onsetCommonMother = "NULL"  # common mother to last one
                # could be a head... might be important here
                nextCommonMother = "NULL"
                lastConstitCommonMother = "NULL"
                interregnumDict[str(repair[7])] += 1
                innerbreak = False

                # i.e. orig, reparandum, interreg, repair, end
                for i in range(len(repair[12])):
                    # (only concerned with first two for now)
                    # key thing we're interested in is the path length of the reparandum onset and of the last consistuent before interruption point versus the others?
                    # i.e. what it is and whether it's higher than the others so far.. would be v interesting.
                    # find out its relative length compared to the rest <S> is
                    # given 10/9 * the next highest, i.e. next length 9, <s>
                    # gets 10
                    pathlengthbucket = repair[12][i]
                    if (i == 0 or i == 1):
                        if i == 1:
                            myDistanceBack = len(pathlengthbucket)
                            if (myDistanceBack == 0):
                                self.errorlog.write("no path lengths (line 412) for " + str(repair[0]) + str(
                                    repair[1]) + str(repair[2]) + str(repair[3]) + str(repair[4]) + "\n")
                                innerbreak = True
                                break
                            reparandumOnsetAddress = pathlengthbucket[0][0]
                            reparandumOnsetPathlength = pathlengthbucket[0][2]
                            onsetCommonMother = pathlengthbucket[0][3]
                            onsetPOSdict[pathlengthbucket[0][1][0]] += 1
                            # i.e. in reparandum there is more than one word
                            if len(pathlengthbucket) > 1:
                                nextCommonMother = pathlengthbucket[1][3]
                            # we need to do a search, there might not be
                            # another constit/ if delete different? for
                            # deletes, these left as NULL
                            else:
                                # search repair and
                                for k in range(i + 2, len(repair[12])):
                                    for d in range(0, len(repair[12][k])):
                                        nextCommonMother = repair[12][k][d][3]
                                        break
                                    if nextCommonMother != "NULL":
                                        break
                            # last constit maybe same as reparandumOnset,
                            # though not neccessarily
                            lastConstitAddress = pathlengthbucket[-1][0]
                            lastConstitPathlength = pathlengthbucket[-1][2]
                            lastConstitCommonMother = pathlengthbucket[-1][3]
                        for pathlength in pathlengthbucket:
                            pathlengths.append(pathlength)
                if (innerbreak == True):
                    continue
                # get the ranking of the reparandum onset.. <S> goes to the top?
                # bubble sort
                # length of turn before onset in morphemes rather than words,
                # so we can look at significance of ranking?
                origLength = len(pathlengths)

                # for weighted rank, first, get distance back for each one:
                for k in range(len(pathlengths) - 1, -1, -1):
                    distanceback = len(pathlengths) - k
                    pathlengths[k].append(distanceback)  # [3]

                swapHappened = True
                while swapHappened == True:
                    swapHappened = False
                    for i in range(len(pathlengths) - 1):
                        length1 = pathlengths[i][2]
                        length2 = pathlengths[i + 1][2]
                        if ((length2 == "<S>" and not length1 == "<S>") or (
                                length1 != "<S>" and int(length2) > int(length1))):
                            pathlengths[i], pathlengths[
                                i + 1] = pathlengths[i + 1], pathlengths[i]
                            swapHappened = True
                # get ranking of reparandumOnset
                # get weightedscore by normalisation (i.e. 10/9 more than
                # second rank (i.e. not the starting constit))
                if len(pathlengths) == 1:
                    # doesn't bother finding it if only one word in
                    relativePathLength = 1
                else:
                    # could have 2 (or more!) <S> in rare cases... make
                    # normaliser be first non s
                    normaliser = 1
                    for pathlength in pathlengths:
                        if not pathlength[2] == "<S>":
                            normaliser = (
                                (float(pathlength[2]) * float(10)) / float(9))
                            break
                    # raw_input(normaliser)
                    for pathlength in pathlengths:
                        if pathlength[2] == "<S>":
                            # i.e. the <S> is given the biggest
                            pathlength[2] = normaliser
                    found = False
                    # pathlengths[0][2] = 1  # <s> always 1, biggest
                    # look down the rankings
                    for i in range(0, len(pathlengths)):
                        # this will return 1 for biggest one
                        myRelativePL = float(
                            pathlengths[i][2]) / float(normaliser)

                        if (pathlengths[i][0] ==
                                reparandumOnsetAddress and found == False):
                            relativePathLength = float(myRelativePL)

                            # raw_input(relativePathLength)
                            found = True
                            break
                        # print(pathlengths[i])
                    # print("")

                # put in appropriate buckets!
                # ADD RAW VALUE # this is for both 1Ps and 3Ps for now
                lengthdict[str(reparandumOnsetPathlength)] += 1
                for bucket in NUMBERBUCKETS:
                    if float(relativePathLength) <= float(bucket[0]):
                        bucket[1] += 1
                        break
                if firstP is True:
                    # ADD RAW VALUE1P
                    firstPlengthdict[str(reparandumOnsetPathlength)] += 1
                    for bucket in firstPNUMBERBUCKETS:
                        if float(relativePathLength) <= float(bucket[0]):
                            bucket[1] += 1
                            break
                else:
                    # ADD RAW VALUE 3P
                    thirdPlengthdict[str(reparandumOnsetPathlength)] += 1
                    for bucket in thirdPNUMBERBUCKETS:
                        if float(relativePathLength) <= float(bucket[0]):
                            bucket[1] += 1
                            break

                # now for origUtt RAW
                # now essentially a table look along top first
                thisbucketfound = False
                for pair in origLengthBucketsRaw:
                    if pair[0] == origLength:
                        for bucket in pair[1]:
                            if str(bucket[0]) == str(
                                    reparandumOnsetPathlength):
                                bucket[1] += 1
                                thisbucketfound = True
                                break
                    if thisbucketfound is True:
                        break

                # now for origUtt RELATIVE
                # now essentially a table look along top first
                thisbucketfound = False
                for pair in origLengthBucketsRelative:
                    if pair[0] == origLength:
                        for bucket in pair[1]:
                            if float(relativePathLength) <= float(bucket[0]):
                                bucket[1] += 1
                                thisbucketfound = True
                                break
                    if thisbucketfound is True:
                        break

                filepair[0].write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(
                    repair[1][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t")
                filepair[0].write(str(origLength) + "\t" + str(myDistanceBack) + "\t" + str(reparandumOnsetPathlength) + "\t" + str(
                    onsetCommonMother) + "\t" + str(nextCommonMother) + "\t" + str(lastConstitPathlength) + "\t" + str(lastConstitCommonMother) + "\t")
                filepair[0].write(str(relativePathLength) + "\n")
                print COUNT
            firstP = False
            filepair[0].close()

        corpus = open(
            "SelfRepairResults\REPAIRCORPUSSTATS22May13.csv.text", "w")
        corpus.write("POS frequency \n")
        for key1, val1 in sorted(onsetPOSdict.items(),
                                 key=itemgetter(0), reverse=True):
            relativefreq = float(val1) / float(3861)
            corpus.write(
                str(key1) + "\t" + str(val1) + "\t" + str(relativefreq) + "\n")
        corpus.write("Interreg occurence \n")
        for key1, val1 in sorted(
                interregnumDict.items(), key=itemgetter(0), reverse=True):
            corpus.write(str(key1) + "\t" + str(val1) + "\n")
        corpus.write("\n OVERALL lengths \n")
        for key1, val1 in sorted(
                lengthdict.items(), key=itemgetter(0), reverse=True):
            corpus.write(str(key1) + "\t" + str(val1) + "\n")
        corpus.write("\n OVERALL buckets \n")
        for pair in NUMBERBUCKETS:
            corpus.write(str(pair[0]) + "\t" + str(pair[1]) + "\n")
        corpus.write("\n 1p lengths \n")
        for key1, val1 in sorted(
                firstPlengthdict.items(), key=itemgetter(0), reverse=True):
            corpus.write(str(key1) + "\t" + str(val1) + "\n")
        corpus.write("\n 1p buckets \n")
        for pair in firstPNUMBERBUCKETS:
            corpus.write(str(pair[0]) + "\t" + str(pair[1]) + "\n")
        """corpus.write("\n 3p lengths \n")
        for key1, val1 in sorted(thirdPlengthdict.items(), key=itemgetter(0), reverse=True):
            corpus.write(str(key1) +  "\t" + str(val1) +  "\n")
        corpus.write("\n 3p buckets \n")
        for pair in thirdPNUMBERBUCKETS:
            corpus.write(str(pair[0]) + "\t" + str(pair[1]) + "\n")
        #now for each orig length!
        """
        corpus.write(
            "\n RAW DISTRIBUTION path length on top, origlength down leftside \n")
        corpus.write(
            "\t <S> \t 1 \t 2 \t 3 \t 4 \t 6 \t 7 \t 8 \t 9 \t 10 \t 11 \t 12 \t 13 \t 14 \t 15")
        for lengthlistpair in origLengthBucketsRaw:
            corpus.write("\n" + str(lengthlistpair[0]))
            for value in lengthlistpair[1]:
                corpus.write("\t" + str(value[1]))
        corpus.write(
            "\n Relative DISTRIBUTION path length on top, origlength down leftside \n")
        corpus.write(
            "\t 0.0<=0.1 \t 0.1<=0.2 \t 0.2<=0.3 \t 0.3<=0.4 \t 0.4<=0.5 \t 0.5<=0.6 \t 0.6<=0.7 \t 0.7<=0.8 \t 0.8<=0.9 \t 0.9<=1.0")
        for lengthlistpair in origLengthBucketsRelative:
            corpus.write("\n" + str(lengthlistpair[0]))
            for value in lengthlistpair[1]:
                corpus.write("\t" + str(value[1]))

        corpus.close()

        print("WE'VE FINISHED")
        #[[self._transcript_index, n],(POS, node),pathlength,commonMother]

    def new_1p_3p(self, trans, utt, mystart, embeddedIn, hasembedded):
        # method to return repair and any embedded ones of 1p or some 3p types (i.e. the 3p ones could be split across their partner's turn../this won't cover same-person Compound Conts which aren't disfluency marked)
        # adds a new repair to the appropriate list
        # embeddedIn essentially says myblah reparandum is in a part of another repair [ blah blah + [  myblah + myblah  ]] myblah is embeddedIn in a REPAIR part of blah
        # hasembedded is the converse, ie the 'blah' in the above jas something
        # embedded in a part of its structure i.e. in its REPAIR

        transcriptFile = trans.swda_filename

        #utterances and indices
        startUtt = utt  # utt where reparandum begins
        startNum = [utt.transcript_index, mystart]

        interregUtt = utt
        # this will change for all, always there even if no interregnum, in
        # which case will be identical to repairNum..
        interregNum = startNum

        # asume repair in same utt as start for 1ps, but this might change for
        # 3ps
        repairUtt = utt
        # this will change for all, as not doing extension repairs here
        repairNum = startNum

        # asume finishes in same utt for 1ps, but this might change for 3ps
        endUtt = utt
        endNum = startNum  # will also obviously change

        # words
        origWords = []
        reparandumWords = []
        interregnumWords = []
        repairWords = []
        endWords = []

        repairFromOther = False

        # treeMaps
        uttTreeMap = self.__treeMapList__.get_treemap(trans, utt)

        # chance there could be more than one utt involved in the repair (i.e.
        # 3p ones), though these will still work on same tree
        startUttTreeMap = uttTreeMap
        # none of these 3 needed for 1ps, this should change for 3ps
        interregUttTreeMap = uttTreeMap
        repairUttTreeMap = uttTreeMap
        endUttTreeMap = uttTreeMap

        # depths (calculated after end)
        origUttDepths = []
        interregnumDepths = []
        reparandumDepths = []
        repairDepths = []
        endDepths = []

        # path lengths (calculated at end)
        origUttPathlengths = []
        reparandumPathlengths = []
        interregnumPathlengths = []
        repairPathlengths = []
        endPathlengths = []

        # interregTypes
        interregTypes = []

        # just for 3ps
        interleavedUtts = []

        # for main string consumption loop
        words = utt.text_words()
        pos = startNum[1]  # word pointer in this utterance
        secondPos = False
        thirdPos = False
        # three-value markers to keep track of string consumption- None (hasn't
        # been reached), True (being consumed) False (consumed)
        reparandumBool = True
        interregnumBool = None
        repairBool = None

        while pos < len(words):
            string = words[pos]
            if string == '[':  # embedded
                pos += 1
                mypos = []
                # we've got a completed reparandum so, (A) [b b + (1)[b b + bb
                # ]] or (B) [b b + (1) b [bb + bb]]
                if reparandumBool == False:
                                                                                # however
                                                                                # (C)
                                                                                # [b
                                                                                # b
                                                                                # +
                                                                                # b
                                                                                # [bb
                                                                                # +
                                                                                # bb]
                                                                                # bb]
                                                                                # so
                                                                                # the
                                                                                # embedded
                                                                                # repairs
                                                                                # repair
                                                                                # repaired
                                                                                # material!
                    # in the middle of an interregnum, unlikely in practice
                    if interregnumBool == True:
                        # marking that there is an embedded repair in the
                        # interregnum, we can later resolve these..
                        hasembedded += "Interregnum."
                        mypos = self.new_1p_3p(
                            trans, utt, pos, embeddedIn + "Interregnum.", "")
                        pos = mypos[1]
                    # i.e. we've completed the reparandum and interregnum
                    elif interregnumBool == False and repairBool == None:
                        # what about if we're at (1) [ bb + {uh} (1) [oo +
                        # nnn]]
                        hasembedded += "Repair."
                        repairNum = [utt.transcript_index, pos + 1]
                        repairBool = True  # it might not be
                        # recursion will retrieve embedded ones too, goes down
                        # one layer, giving another embedding marker
                        mypos = self.new_1p_3p(
                            trans, utt, pos, embeddedIn + "Repair.", "")
                        pos = mypos[1]
                    # we;re in the repair,  got a repair num
                    elif interregnumBool == False and repairBool == True:
                        hasembedded += "Repair."
                        mypos = self.new_1p_3p(
                            trans, utt, pos, embeddedIn + "Repair.", "")
                        pos = mypos[1]
                    # perhaps interreg has not been activated yet... unlikely
                    # to get one now as this would be marked {
                    else:
                        interregNum = [utt.transcript_index, pos + 1]
                        interregnumBool = False  # goes straight to False
                        repairNum = interregNum
                        repairBool = True
                        # interregnum hasn't been activated yet. marking that
                        # there is an embedded repair in the repai we can later
                        # resolve these..
                        hasembedded += "Repair."
                        mypos = self.new_1p_3p(
                            trans, utt, pos, embeddedIn + "Repair.", "")
                        pos = mypos[1]
                # if we've got a [b b [b + b] + b] sit it doesn't change much,
                # but [[ b b + bb ] bb + bbb] or [[ bb + bb] + bb] we need to
                # change the start Num
                elif reparandumBool == True:
                    # if pos-2 == startNum[1]: # maybe always give this..
                    hasembedded += "Reparandum."
                    mypos = self.new_1p_3p(
                        trans, utt, pos, embeddedIn + "Reparandum.", "")
                    pos = mypos[1]

                # now might need to change the utterance and the words.
                if not mypos == []:
                    if not mypos[0] == utt.transcript_index:
                        for myutt in trans.utterances:
                            if myutt.transcript_index > utt.transcript_index and not utt.caller == myutt.caller:
                                thirdPos = True
                            if myutt.transcript_index == mypos[0]:
                                utt = myutt
                                words = utt.text_words()
                                if not thirdPos == True:
                                    secondPos = True
                                break
                        uttTreeMap = self.__treeMapList__.get_treemap(
                            trans, utt)

                        # always true no matter where we are, end treemap is
                        # here for now
                        endUttTreeMap = uttTreeMap
                        if interregnumBool == True or reparandumBool == True:
                            repairUttTreeMap = uttTreeMap
                            if reparandumBool == True:
                                interregUttTreeMap = uttTreeMap

            elif string == ']':
                # should add repair to appropriate list and then return the
                # string position
                # i.e. interregnum and repair never reached, only reparandum
                # terminated.. deletion type
                if interregnumBool is None:
                    interregUtt = utt
                    interregUttTreeMap = self.__treeMapList__.get_treemap(
                        trans, utt)
                    # yes, effectively replacing the ] with +, now need to find
                    # next op to use a +
                    interregNum = [utt.transcript_index, pos]
                    #thirdPos = True

                # i.e. repair not reached yet, interregnum could be over/in
                # progress, still deletion type, look ahead for possible
                # interregnum
                if repairBool is None:
                    repairUtt = utt
                    # repairNum = [utt.transcript_index, pos] # might still have interregna left to add
                    # extend the repair by one word past the repair point to
                    # allow the back-track marker?? f(-1) or s(-1)
                    nextWordFound = False
                    for i in range(pos, len(words)):
                        if i == pos + 2:
                            break  # just look one ahead
                        if words[i] != "--":
                            if "{" in words[i] and words[i][1] != "C":
                                interregTypes.append(words[i][1])
                                interregnumBool = True
                                pos += 1
                                while string[pos] != "}":
                                    pos += 1
                            else:
                                interregnumBool = False
                                if "{" in words[i] and words[i][1] == "C":
                                    pos += 1
                                    # get to end of any bracketed words or
                                    # not...?
                                    while string[pos] != "}":
                                        pos += 1
                                else:
                                    nextWordFound = True
                                    # repairNum now to be inclusive...
                                    repairNum = [utt.transcript_index, pos]
                                    # adding a ] effectively, might be end of
                                    # utt but doesn't matter
                                    endNum = [utt.transcript_index, pos + 1]
                                    break
                        else:
                            pos += 1  # shift along one as "--"

                    # still haven't got a post repair point word, look for next
                    # utterance.. shouldn't have to go beyond that!!
                    if nextWordFound == False:
                        if trans.next_utt_same_speaker(utt) is not None:
                            pos = 0
                            utt = trans.next_utt_same_speaker(utt)
                            words = utt.text_words()
                            thirdPos = True
                            for i in range(pos, len(words)):
                                if i == pos + 2:
                                    break  # just look one ahead
                                if words[i] != "--":
                                    if "{" in words[i] and words[i][1] != "C":
                                        interregTypes.append(words[i][1])
                                        interregnumBool = True
                                        pos += 1
                                        while string[pos] != "}":
                                            pos += 1
                                    else:
                                        interregnumBool = False
                                        if "{" in words[i] and words[i][1] == "C":
                                            pos += 1
                                            # get to end of any bracketed words
                                            # or not...?
                                            while string[pos] != "}":
                                                pos += 1
                                        else:
                                            nextWordFound = True
                                            # repairNum now to be inclusive...
                                            repairNum = [
                                                utt.transcript_index, pos]
                                            # adding a ] effectively, might be
                                            # end of utt but doesn't matter
                                            endNum = [
                                                utt.transcript_index, pos + 1]
                                            break
                                else:
                                    pos += 1  # shift along one as "--"

                    if nextWordFound == False:  # still haven't found it
                        print(
                            "UNRESOLVED possible thirdPos REPAIR BEGINNING UTT" + str(startNum))
                        # raw_input()
                        break  # just crack on and treat it as a normal DEL

                endUtt = utt
                endNum = [utt.transcript_index, pos]
                endUttTreeMap = self.__treeMapList__.get_treemap(trans, endUtt)

                if startNum[0] == endNum[0]:
                    thirdPos = False
                    secondPos = False

                    if "Repair." in embeddedIn:
                        # i.e. there's repaired material in it's reparandum as
                        # it's embedded in a repair
                        hasembedded += "Reparandum."
                    # otherwise hasEmbedding will take care of this- if there's a repair in it's reparandum/interreg/repair this will come up anyway
                    # only might not get noticed - no point saying you're
                    # embedded in a reparandum as you won't know you are yet..

                # this could have changed here due to edit above, due to these
                # being trimming operations a 1p won't go to a 3p..
                if (thirdPos == True or secondPos == True):

                    mytree = None
                    # get all the relevant treemaps, some bug in the system:
                    for myutt in trans.utterances:
                        if myutt.transcript_index == startNum[0]:
                            startUtt = myutt
                        if myutt.transcript_index == interregNum[0]:
                            interregUtt = myutt
                        if myutt.transcript_index == repairNum[0]:
                            repairUtt = myutt
                        if myutt.transcript_index == endNum[0]:
                            endUtt = myutt

                    startUttTreeMap = self.__treeMapList__.get_treemap(
                        trans, startUtt)
                    interregUttTreeMap = self.__treeMapList__.get_treemap(
                        trans, interregUtt)
                    repairUttTreeMap = self.__treeMapList__.get_treemap(
                        trans, repairUtt)
                    ennUttTreeMap = self.__treeMapList__.get_treemap(
                        trans, endUtt)

                    # for d in range(0,len(endUttTreeMap)):
                    #   print(endUttTreeMap[d])
                    #   if len(endUttTreeMap[d][1]) > 0:
                    #      mytree = utt.trees[endUttTreeMap[d][1][0][0]] # gives the treenumber of the repair
                    # TODO, when there are multiple trees in the repair not neccessarily the last one..
                    # need to do search through the words unfort, matching the
                    # start points of each section..
                    if startUttTreeMap.get_last_TreeNumber() is None:
                        self.errorlog.write("NO TREE FOUND FOR repair in  " + startUtt.swda_filename + " utt no." + str(
                            startUtt.transcript_index) + str(startUtt.text_words()))
                        # print(str(startUttTreeMap))
                        # raw_input()
                    # myTreeMaps = list(startUttTreeMap) # this will be added to through concat in 2ps/3ps
                    # gets all the tree lengths in the startUtt
                    #print("startnum " + str(startNum))
                    # print(myTreeMaps)
                    #print("\n\n treemaps so far")

                    treeNumber = 0
                    depths = []  # overall
                    pathlengths = []  # overall
                    for mytree in startUtt.trees:
                        # returns [((47,1), NNP, 2)]
                        depths += startUttTreeMap.get_word_tree_depths(
                            startUtt.transcript_index, mytree, treeNumber)
                        pathlengths += startUttTreeMap.get_word_tree_path_lengths(
                            startUtt.transcript_index, mytree, treeNumber)
                        treeNumber += 1
                    if interregNum[0] > startNum[0]:
                        # might get a different tree..
                        if interregUttTreeMap is None or interregUttTreeMap.get_first_TreeNumber() is None:
                            self.errorlog.write("NO TREE FOUND FOR repair in interregUtt  " + interregUtt.swda_filename + " utt no." + str(
                                interregUtt.transcript_index) + str(interregUtt.text_words()))
                            # print(str(interregUttTreeMap))
                            # raw_input()
                        else:
                            mytree = interregUtt.trees[
                                interregUttTreeMap.get_first_TreeNumber()]
                            interregDepths = interregUttTreeMap.get_word_tree_depths(
                                interregUtt.transcript_index, mytree, 0)
                            interregPathlengths = interregUttTreeMap.get_word_tree_path_lengths(
                                interregUtt.transcript_index, mytree, 0)
                            depths += interregDepths
                            pathlengths += interregPathlengths
                            # myTreeMaps+=interregUttTreeMap
                            #print("adding interreg " + str(interregNum))
                            # print(myTreeMaps)
                    if repairNum[0] > interregNum[0]:
                        if repairUttTreeMap.get_first_TreeNumber() is None:
                            self.errorlog.write("NO TREE FOUND FOR repair in repairUtt " + repairUtt.swda_filename + " utt no." + str(
                                repairUtt.transcript_index) + str(repairUtt.text_words()))
                            # print(str(repairUttTreeMap))
                            # raw_input()
                        else:
                            mytree = repairUtt.trees[
                                repairUttTreeMap.get_first_TreeNumber()]
                        repairdepths = repairUttTreeMap.get_word_tree_depths(
                            repairUtt.transcript_index, mytree, 0)
                        repairPathlengths = repairUttTreeMap.get_word_tree_path_lengths(
                            repairUtt.transcript_index, mytree, 0)
                        depths += repairdepths
                        pathlengths += repairPathlengths
                        # myTreeMaps+=repairUttTreeMap
                        #print("adding repair " + str(repairNum))
                        # print(myTreeMaps)

                    if endNum[0] > repairNum[0]:
                        if endUttTreeMap.get_first_TreeNumber() is None:
                            self.errorlog.write("NO TREE FOUND FOR repair in endUtt " + endUtt.swda_filename + " utt no." + str(
                                endUtt.transcript_index) + str(endUtt.text_words()))
                            # print(str(endUttTreeMap))
                            # raw_input()
                        else:
                            mytree = endUtt.trees[
                                endUttTreeMap.get_first_TreeNumber()]
                        enddepths = endUttTreeMap.get_word_tree_depths(
                            endUtt.transcript_index, mytree, 0)
                        endPathlengths = endUttTreeMap.get_word_tree_path_lengths(
                            endUtt.transcript_index, mytree, 0)
                        depths += enddepths
                        pathlengths += endPathlengths
                        # myTreeMaps+=endUttTreeMap

                        #print("adding end " + str(endNum))
                        # print(myTreeMaps)
                    #print("MYTREEMAP BEFORE ADDING:")
                    # for treemap in myTreeMaps:
                    #    print(treemap)
                    # add words to appropriate bin from concatenated treemaps
                    # and new indices

                    #
                    for i in range(len(startUttTreeMap)):
                        pair = startUttTreeMap[i]
                        if [startUtt.transcript_index, i] < startNum:
                            origWords.append(pair[0])
                        elif [startUtt.transcript_index, i] < interregNum:
                            reparandumWords.append(pair[0])

                    for i in range(len(interregUttTreeMap)):
                        pair = interregUttTreeMap[i]
                        if [interregUtt.transcript_index, i] < interregNum:
                            reparandumWords.append(pair[0])
                        elif [interregUtt.transcript_index, i] < repairNum:
                            interregnumWords.append(pair[0])

                    for i in range(len(repairUttTreeMap)):
                        pair = repairUttTreeMap[i]
                        if [repairUtt.transcript_index, i] < repairNum:
                            interregnumWords.append(pair[0])
                        elif [interregUtt.transcript_index, i] < endNum:
                            repairWords.append(pair[0])

                    for i in range(len(endUttTreeMap)):
                        pair = endUttTreeMap[i]
                        if [endUtt.transcript_index, i] < endNum:
                            repairWords.append(pair[0])
                        else:
                            endWords.append(pair[0])

                    """
                    for i in range(len(myTreeMaps)):
                        pair = myTreeMaps[i]  #treemaps now a pair- oh balls this screws everything up- maybe not
                        #print pair
                        if len(pair[1]) > 0: # only take words with tree mapping?? wise?  yes, gets rid of punct etc..
                            if [startUtt.transcript_index, pair[1]] < startNum:
                                origWords.append(pair[0])
                            elif [interregUtt.transcript_index, pair[0]] < interregNum:
                                reparandumWords.append(pair[0])
                            elif [repairUtt.transcript_index, pair[0]] < repairNum:
                                interregnumWords.append(pair[0])
                            elif [endUtt.transcript_index, pair[0]] < endNum:
                                repairWords.append(pair[0])
                            else: endWords.append(pair[1])
                    """

                    # iterate through each wordpos/depth pair putting into one
                    # of the repair-element bins
                    for n in range(len(depths)):
                        # print(depths[n])
                        # raw_input()
                        if (depths[n][0] < startNum):
                            origUttDepths.append(depths[n])
                        elif (depths[n][0] < interregNum):
                            reparandumDepths.append(depths[n])
                        elif (depths[n][0] < repairNum):
                            interregnumDepths.append(depths[n])
                        elif (depths[n][0] < endNum):
                            repairDepths.append(depths[n])
                        else:
                            endDepths.append(depths[n])

                    myDepths = [origUttDepths, reparandumDepths,
                                interregnumDepths, repairDepths, endDepths]

                    # iterate through each wordpos/pathlength pair putting into
                    # one of the repair-element bins
                    for n in range(len(pathlengths)):
                        # print(pathlengths[n])
                        # raw_input()
                        if (pathlengths[n][0] < startNum):
                            origUttPathlengths.append(pathlengths[n])
                        elif (pathlengths[n][0] < interregNum):
                            reparandumPathlengths.append(pathlengths[n])
                        elif (pathlengths[n][0] < repairNum):
                            interregnumPathlengths.append(pathlengths[n])
                        elif (pathlengths[n][0] < endNum):
                            repairPathlengths.append(pathlengths[n])
                        else:
                            endPathlengths.append(pathlengths[n])

                    myPathLengths = [origUttPathlengths, reparandumPathlengths,
                                     interregnumPathlengths, repairPathlengths, endPathlengths]
                    # need to add something about overalapping speech here I
                    # think..

                    # check for bad stuff/maybe length of tree leaves checked against treemap..
                    #origLeaves = 0
                    #reparandumLeaves = 0
                    #interregnumLeaves = 0
                    #repairLeaves = 0
                    #endLeaves = 0
                   #
                   # for t in range(len(origUtt.trees)):
                   #     origLeaves += len(origUtt.trees[t].leaves)
                   # for t in range(len(reparandumUtt.trees)):
                   #     reparandumLeaves += len(reparandumUtt.trees[t].leaves)
                   # for t in range(len(interregnumUtt.trees)):
                   #     interregnumLeaves += len(interregnumUtt.trees[t].leaves)
                   # for t in range(len(repairUtt.trees[t].leaves)):
                   #     repairLeaves += len(repairUtt.trees[t].leaves)
                   # for t in range(len(endUtt.trees[t].leaves)):
                   #     endLeaves += len(endUtt.trees[t].leaves)

                    if secondPos == True:
                        self.__1plist__.append([transcriptFile, startNum, interregNum, repairNum, endNum, origWords, reparandumWords,
                                                interregnumWords, repairWords, endWords, interregTypes, myDepths, myPathLengths, hasembedded, startUtt.damsl_act_tag()])
                    else:
                        self.__3plist__.append([transcriptFile, startNum, interregNum, repairNum, endNum, origWords, reparandumWords, interregnumWords, repairWords, endWords,
                                                interregTypes, myDepths, myPathLengths, hasembedded, startUtt.damsl_act_tag(), startUtt.damsl_act_tag(), interleavedUtts, True, False, repairFromOther])
                    # print("new3p!")
                    # for element in self.__3plist__[-1]:
                    #    print element
                    # raw_input()

                    # last two Trues in 3p struture says whether it's a split utterance or not, always true here, and whether it's a complete reparandum or not, always false here..
                    # return len(startUtt.text_words()) # risk of duplication, doesn't matter too much as can clean the list afterwords
                    # might be in an embedded one..
                    return endNum
                # it's a 1p// not an 'else' because could have skipped on from
                # previous one if reduced to a 1p from a 3p... bit useless here
                # as if it is a thirdPos will have been added
                else:
                    # put words into appropriate bin:

                    for i in range(len(startUttTreeMap)):
                        # treemaps now a pair- oh balls this screws everything
                        # up- maybe not
                        pair = startUttTreeMap[i]
                        # only take words with tree mapping?? wise?  yes, gets
                        # rid of punct etc..
                        if len(pair[1]) > 0:
                            if i < startNum[1]:
                                origWords.append(pair[0])
                            elif i < interregNum[1]:
                                reparandumWords.append(pair[0])
                            elif i < repairNum[1]:
                                interregnumWords.append(pair[0])
                            elif i < endNum[1]:
                                repairWords.append(pair[0])
                            else:
                                endWords.append(pair[0])

                    # and the new constituent markers (i.e is the next node in
                    # a different subtree to its sisters?)??
                    mytree = None
                    for d in range(startNum[1], len(startUttTreeMap)):
                        if len(startUttTreeMap[d][1]) > 0:
                            if startUttTreeMap[d][1][0][0] is None:
                                pass
                            else:
                                # gives the tree of the repair
                                mytree = startUtt.trees[
                                    startUttTreeMap[d][1][0][0]]
                                mytreenumber = startUttTreeMap[d][1][0][0]
                                break
                    if mytree is None:
                        print("NO TREE FOUND FOR 1p repair in  " + startUtt.swda_filename + " utt no." + str(
                            startUtt.transcript_index) + "from startNum" + str(startNum) + str(startUtt.text_words()))
                        # print(str(startUttTreeMap))
                        # raw_input()

                    treeNumber = 0
                    depths = []  # overall
                    pathlengths = []  # overall
                    for mytree in startUtt.trees:
                        # returns a list of pairs of <wordpos, depth> of each
                        # leaf linked to a word in the tree in question
                        depths += startUttTreeMap.get_word_tree_depths(
                            startUtt.transcript_index, mytree, treeNumber)
                        pathlengths += startUttTreeMap.get_word_tree_path_lengths(
                            startUtt.transcript_index, mytree, treeNumber)
                        treeNumber += 1
                    # put depths into appropriate bin:
                    # iterate through each depth/change pair putting into one
                    # of the repair-element bins
                    for n in range(len(depths)):
                        if depths[n][0] < startNum:
                            origUttDepths.append(depths[n])
                        elif depths[n][0] < interregNum:
                            reparandumDepths.append(depths[n])
                        elif depths[n][0] < repairNum:
                            interregnumDepths.append(depths[n])
                        elif depths[n][0] < endNum:
                            repairDepths.append(depths[n])
                        else:
                            endDepths.append(depths[n])
                    myDepths = [origUttDepths, reparandumDepths,
                                interregnumDepths, repairDepths, endDepths]

                    # put pathlengths in appropriate bins
                    # iterate through each wordpos/pathlength pair putting into
                    # one of the repair-element bins
                    for n in range(len(pathlengths)):
                        # print(pathlengths[n])
                        # raw_input()
                        if pathlengths[n][0] < startNum:
                            origUttPathlengths.append(pathlengths[n])
                        elif pathlengths[n][0] < interregNum:
                            reparandumPathlengths.append(pathlengths[n])
                        elif pathlengths[n][0] < repairNum:
                            interregnumPathlengths.append(pathlengths[n])
                        elif pathlengths[n][0] < endNum:
                            repairPathlengths.append(pathlengths[n])
                        else:
                            endPathlengths.append(pathlengths[n])

                    myPathLengths = [origUttPathlengths, reparandumPathlengths,
                                     interregnumPathlengths, repairPathlengths, endPathlengths]

                    self.__1plist__.append([transcriptFile, startNum, interregNum, repairNum, endNum, origWords, reparandumWords,
                                            interregnumWords, repairWords, endWords, interregTypes, myDepths, myPathLengths, hasembedded, startUtt.damsl_act_tag()])
                    #print("new 1p!")
                    # for element in self.__1plist__[-1]:
                    #    print element
                    # raw_input()
                    # return endNum[1]
                    return endNum
            elif string == '+':
                #interregnumBool = True
                # if
                # else: interregNum = pos
                #repairNum = pos
                reparandumBool = False  # finished reparandum
            # look for interregnum start, but not including CC's "..and...",
            elif (reparandumBool == False and interregnumBool == None):
                if string == "--" and pos + 1 == len(utt.text_words()):
                    pass
                else:
                    interregNum = [utt.transcript_index, pos]
                    if (string[0] == "{" and string[1] != "C"):
                        interregTypes.append(string[1])
                        interregnumBool = True
                        interregNum = [utt.transcript_index, pos]
                    else:
                        # i.e. this is part of the repair instead/there is no
                        # interregnum
                        interregnumBool = False
                        interregNum = [utt.transcript_index, pos]
                        # repair begins here instead
                        repairNum = [utt.transcript_index, pos]
                        repairBool = True
                # mytree = utt.trees[treeMap.getWordMap[pos+1][1][0]] # gives the treenumber of this utterance of the word in question
                # mypos = mytree.leaf_treeposition(utt.trees[[1][1]]) # returns treeposition of the node we're looking at
                # if "INTJ" in TreeMap.get_ancestor_nodes(mytree, mypos): #is this reliable? some CC's and.. could be more fillers than DMs...
                # interregnumbool = True  # i.e. only interjection headed
                # strings within {} added to interregnum?? what about "I mean"
                # {E"
            # won't get mid-repair DMs, which is want we want here
            elif (string == '}' and interregnumBool == True):
                # could be at end of an utt..in which case close of interreg-
                # maybe not for 3ps??
                if ((pos < len(words) -
                     1 and words[pos + 1] != "{") or pos == len(words) - 1):
                    interregnumBool = False
                else:
                    pass
            elif (interregnumBool == False and repairBool == None):
                # and (pos<len(words)-1 and words[pos+1] != "--"):
                if words[pos] != "--":
                    # this word begins the repair
                    repairNum = [utt.transcript_index, pos]
                    repairBool = True
            elif reparandumBool == True:
                pass
                # reparandum.append(string)
            elif interregnumBool == True:
                if (string[0] == "{" and string[1] != "C"):
                    interregTypes.append(string[1])
                # interregnumWords.append(string)
            elif repairBool == True:
                pass
                # repairWords.append(string)
            pos += 1
            # if end of words, start looking at next utt same speaker for 3ps?
            # should only stop when reaching end of transcript if it can't
            # complete repair?/shouldn't happen!
            if pos >= len(words):
                # we might have a second position type annotation, where it's really a first pos
                # for now ignore these- no- don't!
                if (trans.next_utt(utt) is not None and trans.next_utt(
                        utt).caller == startUtt.caller):
                    errormessage = "POSSIBLE SECOND POS REPAIR: START= " + str(startUtt.text_words()) + "NEXT= " + str(
                        trans.next_utt(utt).text_words()) + " " + startUtt.swda_filename + str(startUtt.transcript_index)
                    # self.errorlog.write(errormessage+"\n")
                    yes = "y"
                    if yes == "y":  # make true always now
                        secondPos = True
                        # get the next utt after the interleaved turns same
                        # speaker
                        utt = trans.next_utt(utt)
                        uttTreeMap = self.__treeMapList__.get_treemap(
                            trans, utt)
                        print("my utt= " + str(utt.text_words()))
                        # raw_input()
                        words = utt.text_words()
                        pos = 0
                        # i.e. we haven't got to repair point (still in
                        # reparandum or just finished it with the last string)
                        if interregnumBool is None:
                            interregUtt = utt
                            #interregNum = [pos, utt.transcript_index]
                            interregUttTreeMap = uttTreeMap
                            repairUtt = utt
                            #repairNum = [pos, utt.transcript_index]
                            repairUttTreeMap = uttTreeMap
                            # if we've finished the reparandum, the beginning
                            # of next utterance is the repair
                            if reparandumBool == False:
                                # i.e. 0 of next utt
                                interregNum = [
                                    interregUtt.transcript_index, pos]
                                interregnumBool = True
                        # i.e. we're in the middle of the reparandum, still
                        # haven't got to repair
                        elif (interregnumBool == True and repairBool == None):
                            repairUtt = utt
                            repairUttTreeMap = uttTreeMap
                        # in either case or whether in course of the repair,
                        # the end utt becomes the next utt
                        endUtt = utt
                        endUttTreeMap = uttTreeMap
                        continue
                        # carry on
                    # if we can't deal with it exit this one
                    return len(startUtt.text_words())
                # this is a 3p repair, need to add interim utterances...!
                thirdPos = True
                # keep going until we get all the interleaved utts and the
                # repair one..
                while True:
                    if (trans.next_utt(utt) is not None and trans.next_utt(
                            utt).caller != startUtt.caller):
                        utt = trans.next_utt(utt)
                        uttTreeMap = self.__treeMapList__.get_treemap(
                            trans, utt)
                        #print("interleaved utt= " + str(utt.text_words()))
                        # raw_input()
                        interleavedUtts.append(utt)
                    elif (trans.next_utt(utt) is not None and trans.next_utt(utt).caller == startUtt.caller and trans.next_utt(utt).damsl_act_tag() == "+"):
                        # get the next utt after the interleaved turns same
                        # speaker
                        utt = trans.next_utt(utt)
                        uttTreeMap = self.__treeMapList__.get_treemap(
                            trans, utt)
                        words = utt.text_words()
                        pos = 0

                        # i.e. we haven't got to repair point (still in
                        # reparandum or just finished it with the last string)
                        if interregnumBool is None:
                            interregUtt = utt
                            #interregNum = [pos, utt.transcript_index]
                            interregUttTreeMap = uttTreeMap
                            repairUtt = utt
                            #repairNum = [pos, utt.transcript_index]
                            repairUttTreeMap = uttTreeMap
                            # if we've finished the reparandum, the beginning
                            # of next utterance is the repair
                            if reparandumBool == False:
                                # i.e. 0 of next utt
                                interregNum = [
                                    interregUtt.transcript_index, pos]
                                interregnumBool = True
                        # i.e. we're in the middle of the reparandum, still
                        # haven't got to repair
                        elif (interregnumBool == True and repairBool == None):
                            repairUtt = utt
                            repairUttTreeMap = uttTreeMap
                        # in either case or whether in course of the repair,
                        # the end utt becomes the next utt
                        endUtt = utt
                        endUttTreeMap = uttTreeMap
                        # print("BREAKING!")
                        # raw_input()
                        break
                    else:
                        errormessage = "WARNING: (MAYBE SPLIT UTT) UNRESOLVED 3P REPAIR AT FILE " + str(
                            startUtt.swda_filename) + ", UTT No. " + str(startUtt.transcript_index)
                        print(errormessage)
                        self.errorlog.write(errormessage + "\n")
                        #print("start:"+ str(startUtt.text_words()))
                        #print("interreg:" + str(interregUtt.text_words()))
                        #print("repair:" + str(repairUtt.text_words()))
                        #print("end:" + str(endUtt.text_words()))
                        if len(interleavedUtts) > 0:
                            for interutt in interleavedUtts:
                                pass
                                #print("interleaved:" + str(interutt.text_words()))
                                #print("next utt" + str(trans.next_utt(endUtt).text_words()))
                        # if raw_input("Repair from other y?") == "y":
                        #    repairFromOther == True

                        #
                        break
            #print("\n Repair number " + REPNUM)
            #if len(self.__1plist__)>0: print(self.__1plist__[-1])
            #if len(self.__3plist__)>0: print(self.__3plist__[-1])
            #if utt.transcript_index == 129: raw_input()
            #print("\n current utt = " + str(utt.transcript_index))
            #print("START " + str(startUtt.text_words()) + str(reparandumBool))
            #print("INTERREG  " + str(interregUtt.text_words()) + str(interregnumBool))
            #print("REPAIR " + str(repairUtt.text_words()) + str(repairBool))
            #print("END " + str(repairUtt.text_words()))

        # at the moment don't add this repair if it doesn't terminate in this
        # turn /TODO change can do this now for 3t's/some 3p's
        # TODO need more reliable way of getting position in its iteration..
        return [utt.transcript_index, len(words)]

    # if check is true we're checking..
    def new_1p_3p_NoTreeStuff(
            self, trans, utt, mystart, embeddedIn, hasembedded, check=False):
        """Just return the positions at the moment, for Ngram model. Could also get the lengths across the board too.."""
        # method to return repair and any embedded ones of 1p or some 3p types (i.e. the 3p ones could be split across their partner's turn../this won't cover same-person Compound Conts which aren't disfluency marked)
        # adds a new repair to the appropriate list
        # embeddedIn essentially says myblah reparandum is in a part of another repair [ blah blah + [  myblah + myblah  ]] myblah is embeddedIn in a REPAIR part of blah
        # hasembedded is the converse, ie the 'blah' in the above jas something
        # embedded in a part of its structure i.e. in its REPAIR
        transcriptFile = trans.swda_filename
        transnumber = int(trans.swda_filename[19:23])
        #utterances and indices
        startUtt = utt  # utt where reparandum begins
        startNum = [utt.transcript_index, mystart]

        interregUtt = utt
        # this will change for all, always there even if no interregnum, in
        # which case will be identical to repairNum..
        interregNum = startNum

        # asume repair in same utt as start for 1ps, but this might change for
        # 3ps
        repairUtt = utt
        # this will change for all, as not doing extension repairs here
        repairNum = startNum

        # asume finishes in same utt for 1ps, but this might change for 3ps
        endUtt = utt
        endNum = startNum  # will also obviously change

        # simple function returns the treemap and pos
        uttTreeMap, uttpos = self.getTreeMapAndPos(trans, utt)

        # interregTypes
        interregTypes = []
        # just for 3ps
        interleavedUtts = []

        # for main string consumption loop
        words = utt.text_words()
        pos = startNum[1]  # pos = word postion pointer in this utterance
        # position bools
        secondPos = False
        thirdPos = False
        # repair trackers, 3 values, None=not start, True=Started,
        # False=Completed
        reparandumBool = True
        interregnumBool = None
        repairBool = None

        editterm = False  # whether we're in an edit term phase or not

        while pos < len(words):
            if uttpos is None:  # skip forward over <laughter> turns
                utt = trans.next_utt_same_speaker(utt)
                words = utt.text_words()
                uttTreeMap, uttpos = self.getTreeMapAndPos(trans, utt)
                pos = 0
            string = words[pos]
            # print "top&&&&&&&&&&&&&&&&&&&&&&&"
            # print string
            # print reparandumBool
            # print interregnumBool
            # print repairBool
            if ("{" in string and string[0] == "{" and string[1] != "C"):
                editterm = True
            # decide whether to continue here or not
            # if "Reparandum." in embeddedIn: print "embeddedIn";print pos; print words[pos]; raw_input()
            # look for embedded edited words (chaining/within
            # reparandum/interreg)
            if check == False and ((interregnumBool == True) or (reparandumBool == True)
                                   or ("Reparandum" in embeddedIn)):
                if "Reparandum" in embeddedIn:
                    print string
                self.editedWords[utt.transcript_index].append(pos)

            if string == '[':  # embedded

                pos += 1
                mypos = []
                # we've got a completed reparandum so, (A) [b b + (1)[b b + bb
                # ]] or (B) [b b + (1) b [bb + bb]]
                if reparandumBool == False:
                                                                                # however
                                                                                # (C)
                                                                                # [b
                                                                                # b
                                                                                # +
                                                                                # b
                                                                                # [bb
                                                                                # +
                                                                                # bb]
                                                                                # bb]
                                                                                # so
                                                                                # the
                                                                                # embedded
                                                                                # repairs
                                                                                # repair
                                                                                # repaired
                                                                                # material!
                    # in the middle of an interregnum, unlikely in practice
                    if interregnumBool == True:
                        # marking that there is an embedded repair in the
                        # interregnum, we can later resolve these..
                        hasembedded += "Interregnum."
                        mypos = self.new_1p_3p_NoTreeStuff(
                            trans, utt, pos, embeddedIn + "Interregnum.", "", check)
                        pos = mypos[1]
                    # i.e. we've completed the reparandum and interregnum
                    elif interregnumBool == False and repairBool == None:
                        # nested # what about if we're at (1) [ bb + {uh} (1)
                        # [oo + nnn]]
                        hasembedded += "Repair."
                        # shouldn't do this, as it's been moved on already
                        repairNum = [utt.transcript_index, pos]
                        repairBool = True  # it might not be
                        # recursion will retrieve embedded ones too, goes down
                        # one layer, giving another embedding marker
                        mypos = self.new_1p_3p_NoTreeStuff(
                            trans, utt, pos, embeddedIn + "Repair.", "", check)
                        pos = mypos[1]
                    # we;re in the repair,  got a repair num
                    elif interregnumBool == False and repairBool == True:
                        hasembedded += "Repair."
                        mypos = self.new_1p_3p_NoTreeStuff(
                            trans, utt, pos, embeddedIn + "Repair.", "", check)
                        pos = mypos[1]
                    # perhaps interreg has not been activated yet... unlikely
                    # to get one now as this would be marked {
                    else:
                        # no need to add on
                        interregNum = [utt.transcript_index, pos]
                        interregnumBool = False  # goes straight to False
                        repairNum = interregNum
                        repairBool = True
                        # interregnum hasn't been activated yet. marking that
                        # there is an embedded repair in the repai we can later
                        # resolve these..
                        hasembedded += "Repair."
                        mypos = self.new_1p_3p_NoTreeStuff(
                            trans, utt, pos, embeddedIn + "Repair.", "", check)
                        pos = mypos[1]
                # if we've got a nested [b b [b + b] + b] sit it doesn't change
                # much, but [[ b b + bb ] bb + bbb] or [[ bb + bb] + bb] we
                # need to change the start Num
                elif reparandumBool == True:

                    hasembedded += "Reparandum."
                    mypos = self.new_1p_3p_NoTreeStuff(
                        trans, utt, pos, embeddedIn + "Reparandum.", "", check)
                    pos = mypos[1]
                # now may have jumped forward an utt, so might need to change
                # the utterance, words and treemap, pos returned from embedded
                # in reparandum repair remains the same
                if not mypos == [] and not mypos[0] == utt.transcript_index:
                    for myutt in trans.utterances:
                        if myutt.transcript_index > utt.transcript_index and not utt.caller == myutt.caller:
                            thirdPos = True
                        if myutt.transcript_index == mypos[0]:
                            utt = myutt
                            words = utt.text_words()
                            if not thirdPos == True:
                                secondPos = True
                            uttTreeMap, uttpos = self.getTreeMapAndPos(
                                trans, utt)
                            break

            elif string == ']':
                # should add repair to appropriate list and then give back the
                # string position
                if not reparandumBool == False:
                    print "ERROR- + not observed"
                    print uttpos
                    print pos
                    print startNum
                    self.errorlog.write(
                        "not observed" + str(transcriptFile) + str(utt.transcript_index))
                # i.e. interregnum and repair never reached, only reparandum
                # terminated.. deletion type
                if interregnumBool is None:
                    # yes, effectively replacing the ] with +, now need to find
                    # next op to use a +
                    interregNum = [utt.transcript_index, pos]
                    #thirdPos = True
                if repairBool is None:
                    # repair not reached yet, interregnum could be over/in progress, should be a delete,
                    # should only have delete types
                    print "delete type!!"
                    print uttpos
                    print pos
                    print startNum
                    repairUtt = utt
                    repairNum = [utt.transcript_index, pos]
                    # raw_input()

                endUtt = utt
                endNum = [utt.transcript_index, pos]

                # cancelling function, potentially unneccessary
                if startNum[0] == endNum[0]:
                    thirdPos = False
                    secondPos = False

                if secondPos == True:
                    self.__1plist__.append(
                        [transcriptFile, startNum, interregNum, repairNum, endNum, embeddedIn, startUtt.damsl_act_tag()])
                    return endNum
                elif thirdPos == True:
                    self.__3plist__.append(
                        [transcriptFile, startNum, interregNum, repairNum, endNum, embeddedIn, startUtt.damsl_act_tag()])
                    return endNum
                # it's a 1p// not an 'else' because could have skipped on from
                # previous one if reduced to a 1p from a 3p... bit useless here
                # as if it is a thirdPos will have been added
                else:
                    self.__1plist__.append(
                        [transcriptFile, startNum, interregNum, repairNum, endNum, embeddedIn, startUtt.damsl_act_tag()])
                    return endNum
            elif string == '+':
                reparandumBool = False  # finished reparandum
            elif (reparandumBool == False and interregnumBool == None):
                # search for word after +, look for interregnum start or first
                # repair word, but not including CC's "..and...",
                # we've started an edit term, marked above
                if editterm == True:
                    interregnumBool = True
                    interregNum = [utt.transcript_index, pos]
                elif uttpos[pos] != "null":
                    interregnumBool = False  # no interreg
                    interregNum = [utt.transcript_index, pos]
                    repairNum = [utt.transcript_index, pos]
                    repairBool = True
            # won't get mid-repair DMs, which is want we want here
            elif (interregnumBool == True):
                # only turn off if there's a mapped word that's a non-edit
                # only gets turned on when there's a mappable word
                if editterm == False and uttpos[pos] != "null":
                    interregnumBool = False
                    repairBool = True  # we've started a repairbool
                    # this word begins the repair
                    repairNum = [utt.transcript_index, pos]
            # after an interregnum has been seen
            elif (interregnumBool == False and repairBool == None):
                if uttpos[pos] != "null":  # we have a mapped word
                    # this word begins the repair
                    repairNum = [utt.transcript_index, pos]
                    repairBool = True
            elif reparandumBool == True:  # in the reparandum, keep going
                pass
            elif repairBool == True:
                pass
            if ("}" in string and editterm == True):  # just turn it off here
                editterm = False
            pos += 1
            # if end of words and haven't ended, start looking at next utt same
            # speaker for 3ps? should only stop when reaching end of transcript
            # if it can't complete repair?/shouldn't happen!
            if pos >= len(words):
                # we might have a second position type annotation, where it's really a first pos
                # for now ignore these- no- don't!
                print startNum
                print utt.swda_filename
                print utt.transcript_index
                nextutt = trans.next_utt(utt)
                if (nextutt is not None and nextutt.caller == startUtt.caller):
                    errormessage = "POSSIBLE SECOND POS REPAIR: START= " +\
                        str(startUtt.text_words()) + "NEXT= " + \
                        str(trans.next_utt(utt).text_words()) + " " \
                        + startUtt.swda_filename + \
                        str(startUtt.transcript_index)
                    print errormessage
                    print nextutt.transcript_index
                    # self.errorlog.write(errormessage+"\n")
                    secondPos = True
                    utt = nextutt
                else:
                    print "3p!"
                    thirdPos = True
                    utt = trans.next_utt_same_speaker(utt)
                if utt is not None:
                    words = utt.text_words()
                    uttTreeMap, uttpos = self.getTreeMapAndPos(trans, utt)
                else:
                    print "UNFINISHED REPAIR"
                    print startNum
                pos = 0

        # at the moment don't add this repair if it doesn't terminate in this
        # turn /TODO change can do this now for 3t's/some 3p's
        print "WARNING returning length! " + str(utt.transcript_index)
        # TODO need more reliable way of getting position in its iteration..
        return [utt.transcript_index, len(words)]

    def getTreeMapAndPos(self, trans, utt):
        mytreemap = None
        mypos = None
        if trans.has_trees():
            mytreemap = self.__treeMapList__.get_treemap(trans, utt)
            if not mytreemap is None:
                mypos = mytreemap.get_POS(trans, utt)
        else:  # for POS files
            uttPOSmap = self.__posMapList__.get_POSmap(trans, utt)
            if not uttPOSmap is None:
                mypos = uttPOSmap.get_POS(utt)
        return mytreemap, mypos

    # arguments are all the repair types in order
    def edit_types(self, file1pname, file3pname):
        # bubble sort first, all files!!
        file1p = open(file1pname, "w")
        file3p = open(file3pname, "w")
        lists = self.__1plist__ + self.__3plist__  # concatenate?
        # self.__1plist__ = [] # clear the lists, re-enter them later?
        #self.__3plist__ = []
        swapHappened = True
        print("total no of repairs = " + str(len(lists)))
        raw_input()
        while swapHappened == True:
            swapHappened = False
            for i in range(len(lists) - 1):
                r = lists[i]
                r1 = lists[i + 1]
                rtransnumber = int(r[0][19:23])
                r1transnumber = int(r1[0][19:23])
                # r1 is in lower transcript to r
                if rtransnumber > r1transnumber:
                    self.lists[i] = r1
                    self.lists[i + 1] = r  # swap them
                    swapHappened = True
                elif r[0] == r1[0]:  # we're in the same transcript
                    if r[1][0] == r1[1][0]:  # we're in the same utt
                        # i.e. the start point is after/embedded
                        if r[1][1] > r1[1][1]:
                            lists[i] = r1
                            lists[i + 1] = r  # swap them
                            swapHappened = True

        i = 0
        mytype = ""
        for repair in lists:

            # if (i < 176)  :   # going from the one after the last recorded i
            # <10  repair or in a range  (i < 10 or i > 13) (10-13 inc.)

            # if (repair[1][0]!= 37):
            #    i+=1
            #    continue
            if len(repair) == 15:
                mytype = "1p"
            else:
                mytype = "3p"
            edit = self.editing_type(repair, mytype, i)
            if edit[0] == "quit":
                print("\n Terminating annotation at repair " + str(i) +
                      " : \n " + str(repair[0]) + " start utt " + str(repair[1]))
                break
            if mytype == "3p":
                repair.insert(16, edit)
            else:
                repair.append(edit)
            if mytype == "1p":
                self.__1plist__.append(repair)
                self.print_1p(repair, file1p)  # write each one to the file
            else:
                self.__3plist__.append(repair)
                self.print_3p(repair, file3p)
            i += 1
            print("Repair number " + str(i) + " of" + str(len(lists)))
            raw_input()
            # if i == 1: break

        file1p.close()
        file3p.close()

    def editing_type(self, thisRepair, mytype, number):
        # reparandum, interregnum and repair all have POS tags where possible (all word-pos pairs)
        # Rep, RepSub, RepDel, Sub, SubDel, Del, Ins (7 categories) or Other

        editType = ""
        alledits = []
        # firstly for each repaired utterance, incrementally generate a map of that utterance (embedded repairs may work on the same utterance).
        # 1. sort the repairs by start position or not??  111 [ 111 [222 + 222] + 111]  1) will give fff [fff [fff + fff] + m-3,m-3,m-3]
        # next pass 2) will give fff [fff [ fff + m-3m-3m-3] + m-3m-3m-3- doesn't really matter which way you do it.. interregna can be forward or repaired if you've got I- I mean, and their type in another layer..
        # if you've got deletion... don't think we need to model this as this will come out in wash?? -[and +] what it really means is that the next word goes back with no replacement.. b1
        # we assign values for the whole utterance as to whether

        if (mytype == "3p" or mytype == "1p"):
            # 0 transcriptFile,
            # 10(startUttNum,
            # 11 startNum),
            # 20(interregUttNum,
            # 21 interregNum),
            # 30 (repairUttNum,
            # 31 repairNum),
            # 40(endUttNum,
            # 41 endNum),
            # 5 origWords,
            # 6 reparandumWords,
            # 7 interregnumWords,
            # 8 repairWords,
            # 9 endWords
            # 10 interregTypes,
            # 11 myDepths,
            # 12 myPathLengths
            # 13 emdedding,
            # 14 startUtt.damsl_act_tag()
            # 15 editing types

            origWords = thisRepair[5]
            reparandum = thisRepair[6]
            interregWords = thisRepair[7]
            repair = thisRepair[8]
            endWords = thisRepair[9]
            allDepths = thisRepair[11]

            # give us all the utterances involved?? Need a vector of the context...
            # F permissible grammatical continuation from speaker's last uttered constit if -1
            # FP partial word, still grammatical continuation of last constit, but incomplete
            # RP(-1) repeats part of a previous word sorry [s-]
            # MP(-1) completion of a partial word N words back
            # % end of utt/halt..
            # EDIT SIGNAL..# here or on different level? maybe on different level?
            # S(N)denotes either fresh tree from here or if N>0 fresh start making all words from N words relative to this position (i.e. 3 back if -3) downgraded in terms of discourse status (sort of abandoned)
            # R(N) shows replacement type repair, intended replacement of constituent at pos N detectable via pos matching/syntactic constit
            # I(N) shows potential insert before a word at pos N (i.e. word N back has potential forward dependency on this- NB, is this really incremental in the sense we only know this after the event?)
            # M(N) shows replacement by repetition of a word that far back.. Fairly low level to find dependency in terms of repetition, but to assign the fact that it is a repeat, somewhat different..
            # incrementally predicting the form from the repair onset word onwards could be the task at hand...?
            # Way to do this is, 1. iterate through utterances from first startpoint to last, (this might require a sort) adding "f" unless in the REPAIR-
            # difference is: you need to carry over elements already sorted to the embedded ones... every string with a tree mapping needs one..
            # easy bubble sort?

            # possibly clean for certain disfluencies first?
            # every string in the reparandum must be accounted for to
            # characterize the relation
            myReparandum = ["" for x in range(len(reparandum))]
            myRepair = ["" for x in range(len(repair))]  # this might be empty
            myOrig = ["" for x in range(len(origWords))]
            myInterreg = ["" for x in range(len(interregWords))]
            myEnd = ["" for x in range(len(endWords))]
            # need to make it incremental here. i.e. allow access to previous
            # ones on the list, i.e. scan previous repairs:

            # think this works for both, needs slight alteration
            if mytype == "1p" or mytype == "3p":
                """
                for i in range(number-1,-1,-1): #backwards iter through repairs needs to go back through BOTH lists...
                    yourRepair = self.__lists__[i]
                    print(yourRepair)
                    if (yourRepair[1][0] == thisRepair[1][0]): # in same utt as this one, let's look for crossover
                        mydepths = []
                        for depths in allDepths:
                            for depth in depths:
                                mydepths.append(depth) # reconcatenate depths
                        yourdepths = []
                        for depths in yourRepair[11]: # let's make both lists the same where poss?
                            for depth in depths:
                                yourdepths.append(depth)

                        #makes sure all are single numbers/pairs;
                        for i in range(len(mydepths)-1):
                            if i == len(mydepths)-1: break
                            if mydepths[i][0] == mydepths[i+1][0]: # looking for pairs matching..
                                del mydepths[i]

                        for i in range(len(yourdepths)-1):
                            if i == len(yourdepths)-1: break
                            if yourdepths[i][0] == yourdepths[i+1][0]:
                                del yourdepths[i]

                        #now find the crossover points and populate accordingly
                        yourEditTypes = [] # reconcatenate edittypes
                        for editType in yourRepair[14]:
                            for edits in editType:
                                yourEditTypes += edits
                        #now editTypes and depths should have same length
                        if not len(yourEditTypes) == len(yourdepths):
                            print("WARNING EDIT TYPES AND DEPTHS NOT SAME LENGTH for utt" + str(yourRepair[1]) + " file: " + yourRepair[0] +  " error LINE 764 python")
                            raw_input()
                        editpos = 0
                        youreditpos = 0
                        origBin = True
                        reparandumBin = False
                        interregBin = False
                        endBin = False
                        for mydepth in mydepths:
                            for yourdepth in yourdepths:
                                if mydepth[0] == yourdepth[0]:  # i.e. at same(uttNuM,pos) pair
                                    if origBin == True:
                                        myOrig[editpos] == yourEditTypes[youreditpos] # put edit type in there
                                    elif reparandumBin == True:
                                        myReparandum[editpos] == yourEditTypes[youreditpos]
                                    elif interregBin == True:
                                        myInterreg[editpos] == yourEditTypes[youreditpos]
                                    elif repairBin == True:
                                        myRepair[editpos] == yourEditTypes[youreditpos]
                                    elif repairBin == True:
                                        myEnd[editpos] == yourEditTypes[youreditpos]
                                    else: break # redundant?
                                youreditpos +=1
                            youreditpos = 0
                            editPos+=1
                            if editPos == len(myOrig+myReparandum+myInterreg+myRepair+myEnd):
                                break
                            elif editPos == len(myOrig+myReparandum+myInterreg+myRepair):
                                origBin = False
                                reparandumBin = False
                                interregBin = False
                                repairBin= False
                                endBin = True
                                editPos = 0
                            elif editPos == len(myOrig+myReparandum+myInterreg):
                                origBin = False
                                reparandumBin = False
                                interregBin = False
                                repairBin= True
                                editPos = 0
                            elif editPos == len(myOrig+myReparandum):
                                origBin = False
                                reparandumBin = False
                                interregBin = True
                                editPos = 0
                            elif editPos == len(myOrig):
                                origBin = False
                                reparandumBin = True
                                editPos = 0
                 """
            # now we should have the existing edits for this one possible... we
            # now need to fill in the repair ones../original words that aren't
            # there..
            alphabetUpper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            # we cannot do this easily..., only really helps overlapping
            # repairs
            for i in range(len(myOrig)):
                if (i == 0 and origWords[i][0] in alphabetUpper):
                    if myOrig[i] == "":
                        # fresh start.. not that reliable as common nouns..
                        myOrig[i] = ["s", 0]
                elif myOrig[i] == "":
                    myOrig[i] = ["f", 0]
            # if len(origWords) == 0 and myReparandum
            for i in range(len(myReparandum)):
                if (i == 0 and len(origWords) ==
                        0 and myReparandum[i][0] in alphabetUpper):
                    myReparandum[i] = ["s,", 0]
                elif myReparandum[i] == "":
                    myReparandum[i] = ["f", 0]
            for i in range(len(myInterreg)):
                if myInterreg[i] == "":
                    myInterreg[i] = ["ed", 0]
            # can't assume too much in repair and end edit types.. this is the
            # bit to annotate!

            # else: #repair is equal to or shorter than the reparandum
              # if not Del either straight Rep, RepDel, RepSub, Sub or Other
            if len(repair) == 0:
                for i in range(0, len(myReparandum)):
                    # deletes not really deletes, they are restarts or
                    # abandonments which cause downgrading of discourse status.
                    myReparandum[i] = ["f", 0]
                # if no repair, it's a deletion # need to change this?
                editType = "Del"

                #Rep, RepDel, RepSub, Sub, Insert, SubDel or Other
            i = 0
            j = 0

            repindex = 0

            # look for matching words and matching onsets first:
            for i in range(len(reparandum)):
                for j in range(len(repair)):
                    if (reparandum[i].replace(",", "").replace(".", "") == repair[j].replace(",", "").replace(".", "") or (reparandum[i].replace(",", "").replace(".", "")[-1] == "-" and reparandum[i].replace(
                            ",", "").replace(".", "")[0:len(reparandum[i].replace(",", "").replace(".", "")) - 1] == repair[j].replace(",", "").replace(".", "")[0:len(reparandum[i].replace(",", "").replace(".", "")) - 1])):
                        reparandumconstit = "f"
                        repairconstit = "m"  # i.e. copy
                        # i.e. this is a partial repeat
                        if not reparandum[i].replace(",", "").replace(
                                ".", "") == repair[j].replace(",", "").replace(".", ""):
                            repairconstit += "p"
                        # instead have a -
                        myReparandum[i] = [reparandumconstit, -1]
                        # e.g. (mp, -4) for [F F]{uh}[M(-4) M(-4)]
                        myRepair[j] = [
                            repairconstit, -(j + len(myInterreg) + (len(myReparandum) - i))]
                        repindex += 1
                        break

            if ("" in myReparandum or "" in myRepair):
                pass
            else:
                editType = "Rep"  # must be a rep

                # there are some/all gaps in myReparandum and myRepair, how do we decide what are x's (i.e. inserts/deletes dependent on side and which are r's- replacements?
                # for ambiguous ones present at terminal and do manually?/could
                # do same for semantic relation???? create labels on the fly?
                # Not too bad an idea..

                # if (len(repair)==1 and len(reparandum)==1):
                #    return "Sub.r.r"
                # if (len(reparandum)==2 and len(repair)==1):
                #    if myRepair[0] == "m0":
                # return "RepDel.mx.m"
                # FP partial word, still grammatical continuation of last constit, but incomplete
                # RP(-1) repeats part of a previous word sorry [s-]
                # MP(-1) completion of a partial word N words back
                # % end of utt/halt..
                # EDIT SIGNAL..# E(0) E(-1) continuation of edit signal here or on different level? maybe on different level?
                # S(N)denotes either fresh tree from here or if N>0 fresh start making all words from N words relative to this position (i.e. 3 back if -3) downgraded in terms of discourse status (sort of abandoned)
                # R(N) shows replacement type repair, intended replacement of constituent at pos N detectable via pos matching/syntactic constit
                # I(N) shows potential insert before a word at pos N (i.e. word
                # N back has potential forward dependency on this- NB, is this
                # really incremental in the sense we only know this after the
                # event?)

            origString = ""
            reparandumString = ""
            interregString = ""
            repairString = ""
            endString = ""
            for string in origWords:
                origString += string + " "
            for string in reparandum:
                reparandumString += string + " "
            for string in interregWords:
                interregString += string + " "
            for string in repair:
                repairString += string + " "
            for string in endWords:
                endString += string + " "

            print(
                "\n ANNOTATE REPAIR. \n TYPES: Rep, RepPartial, RepDel, RepSub, Del, Sub, SupRep, SubDel, Insert, InsertPhrase or Other \n CONSTITUENTS: f, fp, m(n), mp(n), r(n), s(n), i(n), d(n), ed(n), t(n)")
            print("\n WORDS = " + origString +
                  "[ " + reparandumString + "+ { " + interregString + "} " + repairString + "] " + endString)
            print("\n EMBEDDED IN= " + thisRepair[12])
            #print("\n editString so far = " + str(myOrig) + "[" + str(myReparandum) + ".{" + str(myInterreg) + "} " + str(myRepair) + " \n words so far = " + str(origWords) + ": [ :" + str(reparandum) + ":+: {" + str(interregWords) + "}" + str(repair) + ":]" + str(endWords))
            myquit = raw_input(
                "For each one: alter or press y to agree then y again to confirm.")
            if myquit == "quit":
                return ["quit", []]
            allwords = [
                origString, reparandumString, interregString, repairString]
            alledits = [myOrig, myReparandum, myInterreg, myRepair]
            allwordssofar = ""
            alleditssofar = []
            happy = "y"
            for k in range(len(alledits)):
                words = allwords[k]
                if k == 0:
                    theType = "orig"
                    allwordssofar += words
                    alleditssofar += alledits[k]
                    continue
                elif k == 1:
                    theType = "reparandum"
                elif k == 2:
                    theType = "interreg"
                elif k == 3:
                    theType = "repair"
                elif k == 4:
                    theType = "end"

                print("words so far : " + str(allwordssofar))
                print("edits so far : " + str(alleditssofar))
                inputString = raw_input(theType + " WORDS: " + str(words) + ". EDIT STRING: " + (
                    str(alledits[k])) + ". length = " + str(len(alledits[k])) + "\n")
                if inputString == "y":
                    inputString = raw_input(
                        "Edit string = " + str(alledits[k]) + "  y?: ")
                    # gives a confirm check,two "Y"s and skips over next bit
                if inputString != "y":   # get new values for editstring
                    while True:
                        print("Edit string = " + inputString)
                        happy = raw_input("happy? y or alter: ")
                        if happy == "y":
                            break
                        else:
                            inputString = happy
                    myalphabet = "fpmrsied"
                    mynums = "-1234567890"
                    print(inputString)
                    for i in range(len(inputString.split())):
                        myed = ""
                        mynum = ""
                        for s in inputString.split()[i]:
                            if s in myalphabet:
                                myed += s
                            if s in mynums:
                                mynum += s
                        print("my num..." + str(mynum) + "...")
                        alledits[k][i] = [myed, int(mynum)]

                allwordssofar += words
                alleditssofar += alledits[k]

            print("reparandum =" + str(myReparandum) +
                  "  repair = " + str(myRepair))
            print(
                "reparandum =" + reparandumString + "  repair = " + repairString)
            print(
                "Rep, RepPartial, RepDel, RepSub, Sub, SubDel, Insert, InsertSub, Other")
            inputString = raw_input("Edit type = " + editType + "  y?")
            if inputString == "y":
                inputString = raw_input("Edit type = " + editType + "  y?")
            if inputString != "y":
                while True:
                    print("edit type = " + inputString)
                    happy = raw_input("happy? y or alter")
                    if happy == "y":
                        break
                    else:
                        inputString = happy

                editType = inputString

        return [editType, alledits]

    ##################################main method##########################

    def self_repairs(self):
        """Pulling out all self-repairs within a single utterance (1p), 2p restarts, \n and those with interleaved material between marked reparandum and repair (3p with continuation turn)."""
        occs = 0
        totalWords = 0
        totalUtts = 0
        totalSpeakers = 0  # can't do it
        totalSpeakersMale = 0
        totalSpeakersFemale = 0
        speakers1019 = 0
        speakers2029 = 0
        speakers3039 = 0
        speakers4049 = 0
        speakers5059 = 0
        speakers6069 = 0
        speakers7079 = 0  # there is 1 older apparantly..
        totalAges = 0
        totalTrees = 0
        totalTopics = 0
        prompts = []
        totalPrompts = 0
        lengthdict = defaultdict(int)
        POSdict = defaultdict(int)
        self.unannotated = 0

        # transcript loop
        for trans in self.corpus.iter_transcripts():
            # the edited words in each transcript, linked to each utt
            self.editedWords = defaultdict(list)

            transnumber = int(trans.swda_filename[19:23])
            if not trans.swda_filename in self.ranges:
                continue
            # if transnumber < self.unannotated_marker[0]: continue #from
            if transnumber > 1210:
                break  # up to
            # if trans.has_trees() is False:
            #    continue
            year = int(str(trans.talk_day)[0:2])
            ageto = year - int(str(trans.from_caller_birth_year)[2:])
            agefrom = year - int(str(trans.to_caller_birth_year)[2:])
            totalAges = totalAges + ageto + agefrom
            totalSpeakers += 2  # some of the callers are the same?
            if not trans.prompt in prompts:
                prompts.append(trans.prompt)
                totalPrompts += 1

            if (ageto > 9 and ageto < 20):
                speakers1019 += 1
            if (agefrom > 9 and agefrom < 20):
                speakers1019 += 1
            if (ageto > 19 and ageto < 30):
                speakers2029 += 1
            if (agefrom > 19 and agefrom < 30):
                speakers2029 += 1
            if (ageto > 29 and ageto < 40):
                speakers3039 += 1
            if (agefrom > 29 and agefrom < 40):
                speakers3039 += 1
            if (ageto > 39 and ageto < 50):
                speakers4049 += 1
            if (agefrom > 39 and agefrom < 50):
                speakers4049 += 1
            if (ageto > 49 and ageto < 60):
                speakers5059 += 1
            if (agefrom > 49 and agefrom < 60):
                speakers5059 += 1
            if (ageto > 59 and ageto < 70):
                speakers6069 += 1
            if (agefrom > 59 and agefrom < 70):
                speakers6069 += 1
            if (ageto > 69 and ageto < 80):
                speakers7079 += 1
            if (agefrom > 69 and agefrom < 80):
                speakers7079 += 1

            if trans.from_caller_sex == "MALE":
                totalSpeakersMale += 1
            if trans.to_caller_sex == "MALE":
                totalSpeakersMale += 1
            if trans.from_caller_sex == "FEMALE":
                totalSpeakersFemale += 1
            if trans.to_caller_sex == "FEMALE":
                totalSpeakersFemale += 1

            translength = len(trans.utterances)
            totalUtts += translength
            count = 0
            # iterating through transcript utterance by utterance
            check = False
            while count < translength:
                utt = trans.utterances[count]
                if utt.damsl_act_tag == "x":
                    count += 1
                    continue
                """
                if self.__treeMapList__.get_treemap(trans, utt) != None:
                    for i in range(len(utt.trees)):
                        mappings = self.__treeMapList__.get_treemap(trans, utt).get_word_tree_path_lengths(utt.transcript_index,utt.trees[i], i)
                        for mapping in mappings:
                            if mapping[2] >0 and mapping[2] <4:
                                self.errorlog.write("\n WARNING odd path length in " +  str(utt.swda_filename) + " " + utt.caller + "." +  str(utt.utterance_index) + "." + str(utt.subutterance_index) + "\n")
                            lengthdict[mapping[2]] +=1
                            POSdict[mapping[1][0]] +=1
                """
                words = utt.text_words()
                cleanwords = utt.text_words(filter_disfluency=True)
                for word in cleanwords:
                    if clean(word) is None:
                        pass
                    else:
                        totalWords += 1
                if check == True:
                    uttTreeMap = self.__treeMapList__.get_treemap(trans, utt)
                else:  # second pass don't have this
                    uttTreeMap = None
                if len(utt.trees) > 0:
                    totalTrees += len(utt.trees)
                pos = 0
                while pos < len(words):
                    string = words[pos]
                    act = utt.damsl_act_tag()

                    missed = False
                    if string == '[' and check == False:
                        # level of embeddng 0 here, this should find all the
                        # self-repairs and the embedded ones so, will resolve
                        # these hopefully..
                        bothpos = self.new_1p_3p_NoTreeStuff(
                            trans, utt, pos + 1, "", "", check)
                        # to deal with embedded:
                        # if bothpos[0]>count:
                        #    print "skipping in main"
                        #    pos = len(words) #if embedded go to next one
                        # pos = bothpos[1] #go to end number
                    # just remove duplicates, though problem is that we might
                    # miss the fact it's embedded
                    elif check == True and not pos in self.editedWords[utt.transcript_index]:
                        # we either haven't reached an edit yet, or it's been missed
                        # pass

                        # only checking mapped words
                        if (not uttTreeMap is None and len(
                                uttTreeMap[pos][1]) > 0):
                            treeIndex = uttTreeMap[pos][1][0][0]
                            leafIndex = uttTreeMap[pos][1][0][1]
                            missed = (not treeIndex is None
                                      and "EDITED" in uttTreeMap.get_ancestor_nodes(uttTreeMap.get_trees(trans)[treeIndex], leafIndex))
                        else:
                            missed = False
                            treeIndex = None
                            leafIndex = None
                        if missed == True:
                            print "missed edited word"
                            # for first automatic step just do one-word delete
                            # structures, obviously more in practice

                            while True:
                                thirdpos = False
                                startUtt = utt
                                # starts bang on the word
                                startNum = [utt.transcript_index, pos]

                                print utt.swda_filename + str(utt.transcript_index)
                                print pos
                                print uttTreeMap[pos]
                                print "\n" + str(utt.text_words())
                                for l in range(0, len(uttTreeMap)):
                                    edited = ""
                                    if l == pos:
                                        edited = "!!E!!"
                                    print str(l) + ": " + str(uttTreeMap[l]) + " " + edited
                                if (transnumber, startNum[0], startNum[1]
                                    ) <= self.unannotated_marker:
                                    print "seen this!!"
                                    break

                                skipCheck = raw_input("go on?")
                                if skipCheck == "n":
                                    break
                                elif skipCheck == "d":
                                    utt.trees[treeIndex].draw()  # raw_input()
                                    skipCheck = raw_input("go on?")
                                    if skipCheck == "n":
                                        break
                                # need a way of scrolling forwards after seeing the
                                # first EDITed word in a sequence
                                startCheck = raw_input(
                                    str(startNum) + " startNum OK?")
                                if startCheck == "n":
                                    startNum = [
                                        utt.transcript_index, int(raw_input("startNum (inc)?"))]
                                # should give use the right thing..
                                interregNum = [
                                    utt.transcript_index, int(raw_input("interreg (inc)?"))]
                                repairNum = [
                                    utt.transcript_index, int(raw_input("repair(inc)?"))]
                                endNum = [
                                    utt.transcript_index, int(raw_input("end(exc)?"))]
                                nextUtt = trans.next_utt_same_speaker(utt)
                                if nextUtt is not None:
                                    print "\n" + str(nextUtt.text_words())
                                    thirdPos = raw_input("3RD?")
                                else:
                                    thirdPos = ""
                                if thirdPos == "y":
                                    nextUtt = trans.next_utt_same_speaker(utt)
                                    nextUttTreeMap = self.__treeMapList__.get_treemap(
                                        trans, nextUtt)
                                    # while not nextUttTreeMap == None:
                                    #    nextUtt = trans.next_utt_same_speaker(nextUtt)
                                    #    nextUttTreeMap = self.__treeMapList__.get_treemap(trans, nextUtt)
                                    for l in range(0, len(nextUttTreeMap)):
                                        print str(l) + ": " + str(nextUttTreeMap[l])
                                    interregNum = parse_list(raw_input(
                                        "interreg (inc) [m,n] (either " + str(nextUtt.transcript_index)or str(utt.transcript_index) + "?"))
                                    repairNum = parse_list(
                                        raw_input("repair(inc)?"))
                                    endNum = parse_list(raw_input("end(exc)?"))
                                    confirm = raw_input("confirm?")
                                    if confirm == "n":
                                        continue
                                    #nextUtt = trans.next_utt_same_speaker(utt)
                                    # pos = 200 #go to end of utt
                                    self.print_1p_NoTreeStuff(
                                        [trans.swda_filename, startNum, interregNum, repairNum, endNum, "", utt.damsl_act_tag()], self.file3p)
                                else:
                                    confirm = raw_input("confirm?")
                                    if confirm == "n":
                                        continue
                                    # might miss nested ones here..
                                    pos = repairNum[1] - 1
                                    self.print_1p_NoTreeStuff(
                                        [trans.swda_filename, startNum, interregNum, repairNum, endNum, "", utt.damsl_act_tag()], self.file1p)
                                self.unannotated_marker = (
                                    transnumber, startNum[0], startNum[1])
                                #self.extras.append([trans.swda_filename, startNum, interregNum, repairNum, endNum, "", utt.damsl_act_tag()])
                                #self.__1plist__.append([trans.swda_filename, startNum, endNum, endNum, endNum, "", startUtt.damsl_act_tag()])

                                #repair = list([trans.swda_filename, startNum, ["x","x"], ["x","x"], ["x","x"], "", utt.damsl_act_tag()])
                                #file1p.write(str(repair[0]) + "\t" + str(repair[1][0]) + "\t" + str(repair[1][1]) + "\t" + str(repair[2][0]) + "\t" + str(repair[2][1]) + "\t" + str(repair[3][0]) + "\t" + str(repair[3][1]) + "\t" + str(repair[4][0]) + "\t" + str(repair[4][1]) + "\t" + str(repair[5]) + "\n")
                                #self.print_1p_NoTreeStuff([trans.swda_filename, startNum, ["x","x"], ["x","x"], ["x","x"], "", utt.damsl_act_tag()],file1p)
                                # trans.utterances[count].trees[treeIndex].draw()
                                # print self.editedWords

                                """
                                #automatic version
                                endUtt = utt
                                endPos = None
                                for i in range(pos+1,len(utt.text_words())):
                                    endPos = i
                                    break
                                if endPos == None:
                                    print "no next word for edited word!"
                                    print startNum
                                    print utt.text_words()
                                    endUtt = trans.next_utt_same_speaker(utt)
                                    if endUtt == None:
                                        "no continuation"
                                        raw_input()
                                        break
                                    endPos = 0
                                endNum = [endUtt.transcript_index,endPos]
                                """

                                self.__1plist__.append(
                                    [trans.swda_filename, startNum, endNum, endNum, endNum, "", startUtt.damsl_act_tag()])
                                # differentiation here not important, only for
                                # detailed annotation later

                                self.unannotated += 1
                                break
                                # self.print_1p_NoTreeStuff([trans.swda_filename, startNum, interregNum, repairNum, endNum, "", utt.damsl_act_tag()],file1p)continue #keep going
                                # raw_input()

                    pos += 1
                count += 1
                # uncomment in annotation mode
                """
                if count == translength and check ==False:
                    print "finished first pass!!"
                    count = 0
                    check = True
                """
        print str(self.unannotated) + " unannotated"
        # raw_input()
        # file1p.close()
        # file3p.close()

        # just 1 and 3 for now
        self.__lists__ = [self.__1plist__, self.__3plist__]
        self.remove_duplicates()  # get rid of duplicated entries
        print "1ps = " + str(len(self.__1plist__))
        print "3ps = " + str(len(self.__3plist__))
        #file1p = open("1pRepairsPathLength.csv.text", "w")
        #file3p = open("3pRepairsPathLength.csv.text", "w")
        # for repair in self.__1plist__:
        #    self.print_1p_NoTreeStuff(repair, file1p)
        # for repair in self.__3plist__:
        #    self.print_3p_NoTreeStuff(repair, file3p)
        print("WORDS = " + str(totalWords))
        print("UTTS = " + str(totalUtts))
        print("MALE = " + str(totalSpeakersMale))
        print("FEMALE = " + str(totalSpeakersFemale))
        if (not totalSpeakersMale + totalSpeakersFemale == totalSpeakers):
            self.errorlog.write("MISMATCH in total speakers!\n")
        print("10-19: " + str(speakers1019))
        print("20-29: " + str(speakers2029))
        print("30-39: " + str(speakers3039))
        print("40-49: " + str(speakers4049))
        print("50-59: " + str(speakers5059))
        print("60-69: " + str(speakers6069))
        print("70-79: " + str(speakers7079))
        #print("(ages sanity check " + str(speakers1019+speakers2029+speakers3039+speakers4049+speakers5059+speakers6069+speakers7079) + " actual " + str(totalSpeakers))
        print("average age = " + str(float(totalAges) / float(totalSpeakers)))
        print("prompts = " + str(totalPrompts))
        print("trees = " + str(totalTrees))
        corpusStatsFile = open(
            "SelfRepairResults\FileCorpusStatsfiles" + self.filename + ".text", "w")
        # how about automatic stats for graphs here?
        # link all categories to those in repair lists then print them side by side?
        # can also get local models here
        for key1, val1 in sorted(
                lengthdict.items(), key=itemgetter(0), reverse=True):
            corpusStatsFile.write(str(key1) + "\t" + str(val1) + "\n")
        for key1, val1 in sorted(
                POSdict.items(), key=itemgetter(0), reverse=True):
            corpusStatsFile.write(
                str(key1) + "\t" + str(val1) + "\t" + str(float(val1) / float(totalWords)) + "\n")
        corpusStatsFile.close()
        assert((speakers1019 + speakers2029 + speakers3039 + speakers4049 +
                speakers5059 + speakers6069 + speakers7079) == totalSpeakers)
        return

    def remove_duplicates(self):
        duplicateNumber = 0
        for mylist in self.__lists__:
            i = 0
            # look for duplicates in next 10 repairs in each list
            while i < len(mylist):
                j = 1
                while j < 11:
                    if i + j == len(mylist):
                        break
                    repair2 = mylist[i + j]
                    if (mylist[i][0] == mylist[i + j][0] and mylist[i][1] == mylist[i + j][1] and mylist[i][2] == mylist[i + j][2]
                            and mylist[i][3] == mylist[i + j][3] and mylist[i][4] == mylist[i + j][4]):
                        del mylist[i + j]
                        # later one always gets deleted, good
                        duplicateNumber += 1
                    else:
                        j += 1
                i += 1
        print("\n" + str(duplicateNumber) + " duplicates removed")
        return

    def corpus_stats(self):
        """Very similar to self_repair method above but does not pull them out, just doing some stats"""
        occs = 0
        totalWords = 0
        totalUtts = 0
        totalSpeakers = 0  # can't do it
        totalSpeakersMale = 0
        totalSpeakersFemale = 0
        speakers1019 = 0
        speakers2029 = 0
        speakers3039 = 0
        speakers4049 = 0
        speakers5059 = 0
        speakers6069 = 0
        speakers7079 = 0  # there is 1 older apparantly..
        totalAges = 0
        totalTrees = 0
        totalTopics = 0
        prompts = []
        totalPrompts = 0
        lengthdict = defaultdict(int)
        POSdict = defaultdict(int)
        self.unannotated = 0

        # transcript loop
        for trans in self.corpus.iter_transcripts():
            # the edited words in each transcript, linked to each utt
            self.editedWords = defaultdict(list)

            transnumber = int(trans.swda_filename[19:23])
            if not trans.swda_filename in self.ranges:
                continue
            # if transnumber < self.unannotated_marker[0]: continue #from
            if transnumber > 1210:
                break  # up to
            # if trans.has_trees() is False:
            #    continue
            year = int(str(trans.talk_day)[0:2])
            ageto = year - int(str(trans.from_caller_birth_year)[2:])
            agefrom = year - int(str(trans.to_caller_birth_year)[2:])
            totalAges = totalAges + ageto + agefrom
            totalSpeakers += 2  # some of the callers are the same?
            if not trans.prompt in prompts:
                prompts.append(trans.prompt)
                totalPrompts += 1

            if (ageto > 9 and ageto < 20):
                speakers1019 += 1
            if (agefrom > 9 and agefrom < 20):
                speakers1019 += 1
            if (ageto > 19 and ageto < 30):
                speakers2029 += 1
            if (agefrom > 19 and agefrom < 30):
                speakers2029 += 1
            if (ageto > 29 and ageto < 40):
                speakers3039 += 1
            if (agefrom > 29 and agefrom < 40):
                speakers3039 += 1
            if (ageto > 39 and ageto < 50):
                speakers4049 += 1
            if (agefrom > 39 and agefrom < 50):
                speakers4049 += 1
            if (ageto > 49 and ageto < 60):
                speakers5059 += 1
            if (agefrom > 49 and agefrom < 60):
                speakers5059 += 1
            if (ageto > 59 and ageto < 70):
                speakers6069 += 1
            if (agefrom > 59 and agefrom < 70):
                speakers6069 += 1
            if (ageto > 69 and ageto < 80):
                speakers7079 += 1
            if (agefrom > 69 and agefrom < 80):
                speakers7079 += 1

            if trans.from_caller_sex == "MALE":
                totalSpeakersMale += 1
            if trans.to_caller_sex == "MALE":
                totalSpeakersMale += 1
            if trans.from_caller_sex == "FEMALE":
                totalSpeakersFemale += 1
            if trans.to_caller_sex == "FEMALE":
                totalSpeakersFemale += 1

            translength = len(trans.utterances)
            totalUtts += translength
            count = 0
            # iterating through transcript utterance by utterance
            check = False
            while count < translength:
                utt = trans.utterances[count]
                if utt.damsl_act_tag == "x":
                    count += 1
                    continue
                """
                if self.__treeMapList__.get_treemap(trans, utt) != None:
                    for i in range(len(utt.trees)):
                        mappings = self.__treeMapList__.get_treemap(trans, utt).get_word_tree_path_lengths(utt.transcript_index,utt.trees[i], i)
                        for mapping in mappings:
                            if mapping[2] >0 and mapping[2] <4:
                                self.errorlog.write("\n WARNING odd path length in " +  str(utt.swda_filename) + " " + utt.caller + "." +  str(utt.utterance_index) + "." + str(utt.subutterance_index) + "\n")
                            lengthdict[mapping[2]] +=1
                            POSdict[mapping[1][0]] +=1
                """
                words = utt.text_words()
                cleanwords = utt.text_words(filter_disfluency=True)
                for word in cleanwords:
                    if clean(word) is None:
                        pass
                    else:
                        totalWords += 1
                if check == True:
                    uttTreeMap = self.__treeMapList__.get_treemap(trans, utt)
                else:  # second pass don't have this
                    uttTreeMap = None
                if len(utt.trees) > 0:
                    totalTrees += len(utt.trees)
                pos = 0
                while pos < len(words):
                    string = words[pos]
                    act = utt.damsl_act_tag()

                    missed = False

                    pos += 1
                count += 1
                # uncomment in annotation mode
                """
                if count == translength and check ==False:
                    print "finished first pass!!"
                    count = 0
                    check = True
                """
        print str(self.unannotated) + " unannotated"
        # raw_input()
        # file1p.close()
        # file3p.close()

        # just 1 and 3 for now
        self.__lists__ = [self.__1plist__, self.__3plist__]
        self.remove_duplicates()  # get rid of duplicated entries
        print "1ps = " + str(len(self.__1plist__))
        print "3ps = " + str(len(self.__3plist__))
        #file1p = open("1pRepairsPathLength.csv.text", "w")
        #file3p = open("3pRepairsPathLength.csv.text", "w")
        # for repair in self.__1plist__:
        #    self.print_1p_NoTreeStuff(repair, file1p)
        # for repair in self.__3plist__:
        #    self.print_3p_NoTreeStuff(repair, file3p)
        corpusStatsFile = open(
            "SelfRepairResults\FileCorpusStatsfiles" + self.filename + ".csv.text", "w")
        print("WORDS = " + str(totalWords))
        corpusStatsFile.write("WORDS = \t" + str(totalWords) + "\n")
        print("UTTS = " + str(totalUtts))
        corpusStatsFile.write("UTTS = \t" + str(totalUtts) + "\n")
        print("MALE = " + str(totalSpeakersMale))
        corpusStatsFile.write("MALE = \t" + str(totalSpeakersMale) + "\n")
        print("FEMALE = " + str(totalSpeakersFemale))
        corpusStatsFile.write("FEMALE = \t" + str(totalSpeakersFemale) + "\n")
        print("10-19: " + str(speakers1019))
        corpusStatsFile.write("10-19: \t" + str(speakers1019) + "\n")
        print("20-29: " + str(speakers2029))
        corpusStatsFile.write("20-29: \t" + str(speakers2029) + "\n")
        print("30-39: " + str(speakers3039))
        corpusStatsFile.write("30-39: \t" + str(speakers3039) + "\n")
        print("40-49: " + str(speakers4049))
        corpusStatsFile.write("40-49: \t" + str(speakers4049) + "\n")
        print("50-59: " + str(speakers5059))
        corpusStatsFile.write("50-59: \t" + str(speakers5059) + "\n")
        print("60-69: " + str(speakers6069))
        corpusStatsFile.write("60-69: \t" + str(speakers6069) + "\n")
        print("70-79: " + str(speakers7079))
        corpusStatsFile.write("70-79: \t" + str(speakers7079) + "\n")
        #print("(ages sanity check " + str(speakers1019+speakers2029+speakers3039+speakers4049+speakers5059+speakers6069+speakers7079) + " actual " + str(totalSpeakers))
        print("average age = " + str(float(totalAges) / float(totalSpeakers)))
        corpusStatsFile.write(
            "average age = \t" + str(float(totalAges) / float(totalSpeakers)) + "\n")
        print("prompts = " + str(totalPrompts))
        corpusStatsFile.write("prompts = \t " + str(totalPrompts) + "\n")
        print("trees = " + str(totalTrees))
        corpusStatsFile.write("trees = \t" + str(totalTrees) + "\n")

        # how about automatic stats for graphs here?
        # link all categories to those in repair lists then print them side by side?
        # can also get local models here
        for key1, val1 in sorted(
                lengthdict.items(), key=itemgetter(0), reverse=True):
            corpusStatsFile.write(str(key1) + "\t" + str(val1) + "\n")
        for key1, val1 in sorted(
                POSdict.items(), key=itemgetter(0), reverse=True):
            corpusStatsFile.write(
                str(key1) + "\t" + str(val1) + "\t" + str(float(val1) / float(totalWords)) + "\n")
        corpusStatsFile.close()
        assert((speakers1019 + speakers2029 + speakers3039 + speakers4049 +
                speakers5059 + speakers6069 + speakers7079) == totalSpeakers)
        assert(totalSpeakersMale + totalSpeakersFemale == totalSpeakers)
        return

    def dialogue_acts(self, percentage=True):
        d = defaultdict(int)
        d1 = defaultdict(int)
        corpus = CorpusReader('../swda')
        file = open('DActs.text', 'w')
        file1 = open('DActsCorpus.text', 'w')
        totalwords = 0
        totalrepairs = 0
        # normalising
        for utt in corpus.iter_utterances(display_progress=False):
            # d[utt.damsl_act_tag()] += len(utt.text_words(filter_disfluency=True)) # normalising over words
            #totalwords +=len(utt.text_words(filter_disfluency=True))

            d[utt.damsl_act_tag()] += 1  # normalising over acts
        for list in self.__repairlist__:
            d1[list[3]] += 1
            totalrepairs += 1
        for key1, val1 in sorted(d1.items(), key=itemgetter(1), reverse=True):
            # normalising by dividing by number of words per act tag
            # indent th next three lines!
            if percentage:
                for key, val in sorted(
                        d.items(), key=itemgetter(1), reverse=True):
                    # Getting raw acts over entire corpus
                    file1.write(key + "," + str(val))
                    if key == key1:
                        # replacing value with normalized value
                        # float(val1/len(self.__repairlist__)) * 100
                        d1[key] = float(val1) / val
        for key1, val1 in sorted(d1.items(), key=itemgetter(1), reverse=True):
            # print to csv format
            file.write(key1 + "," + str(val1) + "," +
                       self.forward_backward(key1) + "," + str(d[key1]) + "\n")
            print key1, val1
        file.close()
        file1.close()

    def dialogue_act_utterance(self, string, double=False):
        # retrieves the utterance texts for particular dialogue act and turn
        # either side
        file = open('DAutts3t.text', 'w')
        corpus = CorpusReader('../swda')
        occs = 0
        for trans in corpus.iter_transcripts():
            translist = trans.utterances
            translength = len(translist)
            count = 0
            # iterating through transcript utterance by utterance
            while count < translength:
                utt = translist[count]
                if count <= 1:
                    count += 1
                    # ignore first two for now. i.e. can't get a 3t repair here
                    # now
                    continue
                if utt.caller == trans.utterances[count - 1].caller:
                    count += 1
                    continue  # only look for different speakers for now

                if (utt.damsl_act_tag() == string):
                    if (double):
                        if (count >= (translength - 1) or utt.caller !=
                                trans.utterances[count + 1].caller or trans.utterances[count + 1].damsl_act_tag() != string):
                            # i.e. skip this one if we're looking for doubles
                            # and the following utterance doesn't have the same
                            # tag
                            count += 1
                            continue
                    print '------------------'
                    file.write("------------------------- \n")
                    if count > 0:  # give if there
                        print trans.utterances[count - 2].damsl_act_tag(), " ", trans.utterances[count - 2].caller, " : ", trans.utterances[count - 2].text_words()
                        file.write(trans.utterances[count - 2].damsl_act_tag() + " " + trans.utterances[
                                   count - 2].caller + " : " + str(trans.utterances[count - 2].text_words()) + "\n")
                        print trans.utterances[count - 1].damsl_act_tag(), " ", trans.utterances[count - 1].caller, " : ", trans.utterances[count - 1].text_words()
                        file.write(trans.utterances[count - 1].damsl_act_tag() + " " + trans.utterances[
                                   count - 1].caller + " : " + str(trans.utterances[count - 1].text_words()) + "\n")

                    # give target utterance
                    print utt.damsl_act_tag(), " ", utt.caller, " : ", utt.text_words()
                    file.write(
                        utt.damsl_act_tag() + " " + utt.caller + " : " + str(utt.text_words()) + "\n")

                    # give following utterance if there
                    if (count < (translength - 1)):
                        print trans.utterances[count + 1].damsl_act_tag(), " " + trans.utterances[count + 1].caller, " : ", trans.utterances[count + 1].text_words()
                        file.write(trans.utterances[count + 1].damsl_act_tag() + " " + trans.utterances[
                                   count + 1].caller + " : " + str(trans.utterances[count + 1].text_words()) + "\n")
                    occs += 1
                count += 1
        print occs, "occurences"
        file.write(str(occs) + "occurences")

    def dialogue_act_repairs(self, string):
        # retrieves the utterance texts for particular dialogue act
        for list in self.__repairlist__:
            if list[3] == string:
                print list[5].text_words(filter_disfluency=False)

    def forward_backward(self, DA):
        for string in self.forward_acts:
            if string == DA:
                return "f"
        for string in self.backward_acts:
            if string == DA:
                return "b"
        return "o"


if __name__ == '__main__':
    s = SelfRepair()
    #reparandum = [("there","EX"),("were","VBD")]
    #repair = [("they","PRP")]
    #continuation = [("a","DT")]
    #repair = [("You","NNP"),("really","RB"),("like","VP"),("him","NP")]
    #reparandum = [("Y-","NNP"),("like","VP"),("john","NN")]
    #repair = [("Y-","NNP"),("like","VP"),("and","cc"),("I","RB"),("like","VP"),("I","RB"),("like","VP"),("john","NN")]
    # graph_viz_repair(classify(reparandum,repair,[("","")]),reparandum,repair,continuation)
