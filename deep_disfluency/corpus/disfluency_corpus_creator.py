# -*- coding: utf-8 -*-
"""
Class for creating various disfluency marked-up and
related corpora from various required input files.

Input:
1. SWDA-style utt.csv files in a directory.
2. POSmap/Treemaps as created by tree_pos_map_writer.py
3. (if available) stand-off repair annotation files (.csv files)
4. (if available) spelling normalization for a given language,
in the 'resource' folder by default

Output (available through the various corpus creation options):
1. Disfluency detection corpus - marked up according to gold
standard repair/edit term mark-up. An EasyRead verison of this
is also available where the tags are in-line.
2. Clean corpus- a corpus which can be used for language model
training which is cleaned of disfluency  (i.e. reparanda,
interregna and other edit terms removed)
3. Edit term corpus- a corpus of only consecutive edit term words.
4. Disfluency Type Analysis- information on the types of repairs,
edit terms found in the corpus
5. Tree corpus/analysis- a corpus of incremental PTB style
tree contexts for each word.
"""
import argparse
import os
import re
import csv
import string
import sys
from copy import deepcopy
from glob import iglob
from collections import defaultdict, Counter

from swda import CorpusReader
from tree_pos_map import TreeMapCorpus, POSMapCorpus
from tree_pos_map_writer import POSMapWriter, TreeMapWriter
from repair import Repair
from util import classify_repair, graph_viz_repair
from util import disf_tags_from_easy_read
from util import strip_disf_tags_from_easy_read
from util import easy_read_disf_format
from util import get_edit_terms_from_easy_read
from util import detection_corpus_format
from util import detection_corpus_format_from_easy_read
from util import easy_read_format_from_detection_corpus
from util import verify_disfluency_tags
from util import orthography_normalization
from util import find_repair_end, find_repair_end_in_previous_utts
from util import find_delete_interregna_and_repair_onsets
from util import remove_non_edit_interregna
from util import find_repair_ends_and_reclassify
from util import find_interregna_and_repair_onsets
from util import remove_repairs, remove_partial_words
from spelling_dictionary import SpellingDict


def annotate_relaxed_repairs(tag_list, repair_list):
    """An ad hoc method using 'relaxed' repair information
    about how many repair onsets there are in this utt,
    not their precise locations.
    Modifies tag_list in place with the new 'fake' tags
    """
    repairIndex = 0
    for i in range(0, len(tag_list)):
        if repairIndex < len(repair_list):
            ID = repair_list[repairIndex]
            # mark the fake reparandum
            tag_list[i] = tag_list[i] + '<rms id="{}"/>'.format(ID)
            tag_list[i + 1] = tag_list[i + 1] + '<rps id="{}"/>'.format(ID) +\
                '<rpnsub id="{}"/>'.format(
                    ID)  # and fake 1 word substitution repair
            repairIndex += 1
    return


class DisfluencyCorpusCreator:
    """An object which writes disfluency related corpora when
    the makeCorpus method is called.
    """

    def __init__(self, path, metadata_path=None, mapdir_path=None,
                 partial=True, d_acts=True, annotation_files=None,
                 annotation_files_relaxed=None, error_log=None, lang='en'):
        data_warning = "data folder not in raw_data, please move it in there!"
        if not os.path.exists(path):
            print data_warning, path
            quit()

        if error_log:
            self.errorlog = open(error_log, "w")
        else:
            self.errorlog = None

        self.partial_words = partial  # turn off/on
        self.d_acts = d_acts

        self.corpus = CorpusReader(path, metadata_path)

        try:
            self.__treemaplist__ = TreeMapCorpus(True, self.errorlog,
                                                 mapdir=mapdir_path)
        except BaseException:
            self.__treemaplist__ = None
            print "WARNING no treemap files"
        try:

            self.__POSmaplist__ = POSMapCorpus(True, self.errorlog,
                                               mapdir=mapdir_path)
        except BaseException:
            self.__POSmaplist__ = None
            print "WARNING no posmap files"

        # get spelling dict for language
        self.lang = lang
        self.spelling_dict = SpellingDict(
            os.path.dirname(os.path.realpath(__file__)) + '/resource/' +
            'spelling_' + lang + ".text")

        # default to the main training files
        self.annotationFiles = annotation_files
        self.annotationFilesRelaxed = annotation_files_relaxed

        # dict of list of all repairs
        # keys reparandum start, values lists of positions of
        # the different repair structures starting there,
        # embedded and type (1p or 3p)
        self.dRepairs = defaultdict(list)
        self.dEditTerms = defaultdict(list)  # all the edit terms positions
        self.dRelaxedRepairs = None

        self.ranges = []

    def read_in_self_repair_files(self):
        """Read in self repair annotations from files
        into a dictionary used by corpus creation methods.
        """
        rows = []
        repairID = 0  # unique ID for each repair annotated
        if self.annotationFiles:
            for myfile in self.annotationFiles:
                counter = 0
                with open(myfile, 'r') as g:
                    print myfile
                    myreader = csv.reader(g, delimiter='\t')
                    for a, b, c, d, e, f, g, h, i, j, k in myreader:
                        # if not a in self.ranges:
                        #    continue
                        counter += 1
                        rows.append([a, b, c, d, e, f, g, h, i, j, k])
                        repairID += 1
                        try:
                            self.dRepairs[a + ":" + str(b)]\
                                .append(Repair([int(b), int(c)],
                                               [int(d), int(e)],
                                               [int(f), int(g)],
                                               [int(h), int(i)],
                                               repairID))
                        except BaseException:
                            warning = "invalid repair" + \
                                " ".join([a, b, c, d, e, f, g, h, i, j, k])
                            print warning
                            if self.errorlog:
                                print>>self.errorlog, warning

        else:
            print "No normal style repair annotations"
            return
        print str(len(rows)) + " total repairs loaded"
        return

    def read_in_relaxed_self_repair_files(self):
        """Read in relaxed (per-turn) self repair annotations
        from files into a dictionary used by corpus creation methods.
        """
        self.dRelaxedRepairs = defaultdict(list)
        repairID = 0  # unique repair ID for each repair
        if self.annotationFilesRelaxed:
            for myfile in self.annotationFilesRelaxed:
                with open(myfile, 'r') as f:
                    reader = csv.reader(f, delimiter=',')
                    for a, b, c, d, e, f, g, h, i in reader:
                        try:
                            for _ in range(0, int(i)):
                                repairID += 1
                                self.dRelaxedRepairs[c + ":" + d]\
                                    .append(repairID)
                        except BaseException:
                            warning = "invalid relaxed repair" + " "\
                                .join([a, b, c, d, e, f, g, h, i])
                            print warning
                            if self.errorlog:
                                print>>self.errorlog, warning
        print repairID, "relaxed repairs loaded"
        return

    def write_test_corpus(self, filename, ranges=None, partial=True,
                          writeFile=True, edit_terms_marked=True,
                          edit_term_labels=["F", "D", "E"],
                          dodgy=["NNP", "NNPS", "CD", "LS", "SYM", "FW"],
                          filled_pauses=["um", "uh"], debug=False):
        """
        Method to write a disfluency test corpus that can be used by
        any disfluency detection system for training/testing.
        Outputs a corpus of words, pos tags (if available) and the
        repair and edit term tags from the annotations (repairs)
         or marked inline (edit terms).
        """
        overallWordsList = []  # all lists of words
        overallPOSList = []  # all lists of corresponding POS tags
        overallTagList = []  # all lists of corresponding Disfluency Tags
        # all list of corresponding index tags back to the original
        # transcription
        overallIndexList = []
        uttList = []

        totalRepairs = 0
        wordCount = 0
        numberTrans = 0

        errors = 0
        warnings = 0

        # TODO test/debug stuff
        my_test_dodgy = "4646:A:10"
        # WARNING CHANGE BACK:
        #ranges = ["4646"]

        for trans in self.corpus.iter_transcripts():
            # check to make sure transcript is in the desired ranges, if there
            # are ranges
            if ranges and not trans.conversation_no in ranges:
                continue
            # FIFO stack of repair objects can be from any conversation
            # participant
            repairStack = []

            numberTrans += 1
            # iterate over transcript utterances
            count = 0
            while count < len(trans.utterances):
                utt = trans.utterances[count]
                # get the tree maps or pos map to get the pos tags for the utt
                mytreemap = None
                myPOSmap = None
                if trans.has_trees():
                    mytreemap = self.__treemaplist__.get_treemap(trans, utt)
                    if mytreemap:
                        mypos = mytreemap.get_POS(trans, utt)
                    else:
                        count += 1
                        continue
                else:  # for POS files
                    myPOSmap = self.__POSmaplist__.get_POSmap(trans, utt)
                    if myPOSmap:
                        mypos = myPOSmap.get_POS(utt)
                    else:
                        count += 1
                        continue

                # for this utterance, we have the words, pos, disfluency tags,
                # and index to the original word positions
                wordList = []
                POSList = []
                disfluencyTagList = []
                indexList = []

                uttReference = trans.conversation_no + \
                    ":" + str(utt.transcript_index)
                test_dodgy = trans.conversation_no + ":" + \
                    str(utt.caller) + ":" + str(utt.transcript_index)
                # the repairs with their reparanda beginning in this utterance
                repairPoints = self.dRepairs.get(uttReference)

                # get repair objects attached to this start if any, and add
                # their caller
                if repairPoints:
                    for repair in repairPoints:
                        repair.caller = utt.caller
                        repairStack.append(repair)

                if debug:
                    print "(1) repair points for " + uttReference + " " + \
                        str(count) + " " + str(utt.transcript_index)
                    print[repair.to_string() for repair in repairStack]

                MD = False  # boolean, whether middle of an edit term or not
                # iterate over the words, creating the appropriate disfluency
                # tags
                for i in range(0, len(utt.text_words())):

                    word = utt.text_words()[i].replace(',', '').\
                        replace(".", "").replace("?", "").replace(
                        "(", "").replace(")", "").replace('"', "").\
                        replace("'", "").replace("/", "").lower()
                    pos = mypos[i]
                    # assuming fluent, add disfluency tags as we go along
                    disfluency_tag = ""

                    # check to see if we're in an edit term according to the
                    # edit_term_labels
                    if "{" in word and \
                            utt.text_words()[i][1] in edit_term_labels:
                        MD = True
                    if MD == True and word == "}":
                        MD = False

                    # pop stack of completed repairs
                    # check whether any completed repairs remaining in repair
                    # stack
                    repairStackPopped = False
                    while repairStackPopped == False:
                        repairStackPopped = True
                        for rs in range(0, len(repairStack)):
                            repair = repairStack[rs]
                            tag = repair.in_segment([utt.transcript_index, i])
                            # any finished tags go that have been completed
                            if tag is None and repair.complete == True:
                                repairStack.pop(rs)
                                repairStackPopped = False
                                break

                    if debug:
                        print "(2) repair points after POP at word", i
                        print[repair.to_string() for repair in repairStack]
                        if test_dodgy == my_test_dodgy:
                            raw_input()

                    if (trans.has_trees() == True and mytreemap[i][1] == [])\
                            or (trans.has_trees() == False and
                                myPOSmap[i][1] == [])\
                            or (word == ""):
                        continue  # only add words with a pos tag

                    # If we get here we're definitely adding the word,pos,tag
                    word, pos = orthography_normalization(
                        word, pos, self.spelling_dict, self.lang)
                    if pos in dodgy:
                        # to differentiate letters like 'I' from other meanings
                        # principally
                        word = "$unc$" + word

                    # now iterate through the remaining repairs
                    # in the stack and add appropriate disfluency tags
                    # for this word
                    # or search for previous utterances
                    if debug:
                        print "word added", i, word
                        print "repair stack", [repair.to_string()
                                               for repair in repairStack]
                        if test_dodgy == my_test_dodgy:
                            raw_input()
                    for rs in range(0, len(repairStack)):
                        repair = repairStack[rs]
                        if debug:
                            print repair.to_string()
                        tag = repair.in_segment([utt.transcript_index, i])
                        if not repair.caller == utt.caller:
                            continue
                        if tag == "o":  # original utterance, not reached yet?
                            continue
                        # i.e. it was never repairStackPopped, gets rid of one
                        # word partials and edit term repairs etc..
                        elif (tag == "rp" or tag == "i" or tag is None) \
                                and repair.reparandum == False:
                            # pops it off but doesn't annotate it
                            repair.complete = True
                            warning = "ANNOTATION ERROR: Reparandum \
                            never parsed. \n" + utt.swda_filename + \
                                " " + uttReference + " " + \
                                repair.to_string() + "\n" + \
                                str(utt.text_words())
                            if debug:
                                print warning
                            errors += 1
                            if self.errorlog:
                                self.errorlog.write(warning)
                        elif tag == "rm":
                            if repair.reparandum == False:
                                repair.reparandum = True  # pop it
                                tag += "s"  # indicates start
                            # no tags for classification
                            repair.reparandumWords.append((word, pos))
                        elif tag == "i":
                            pass
                        elif tag == "rp":
                            # only adding first rp constit, and interreg
                            if repair.repair == False:
                                repair.repair = True
                                tag += "s"  # indicates start of repair
                            # no tags for classification
                            repair.repairWords.append((word, pos))
                        elif tag is None and repair.complete == False:
                            if debug:
                                print "incomplete tag"
                            # Have gone past last word of the repair but
                            # not found completion
                            # Need to backwards search to find it and mark
                            # its type
                            # Can involve searching back before the current
                            # utterance in third pos/intra-utterance repairs
                            if repair.repair == False:  # delete
                                disfluency_tag += '<rps id="{}"/>'.format(
                                    repair.repairID) + \
                                    '<rpndel id="{}"/>'.format(repair.repairID)
                                repair.complete = True
                            else:
                                # if not delete, search back for last repair
                                # word
                                repair.complete = find_repair_end(
                                    repair, disfluencyTagList)
                                if not repair.complete:
                                    # if not found in this utt, search back
                                    try:
                                        find_repair_end_in_previous_utts(
                                            repair, overallTagList, uttList)
                                    except BaseException:
                                        warning = "ANNOTATION ERROR: \
                                        Start of repair not found. \n" + \
                                            utt.swda_filename + " " +\
                                            uttReference + " " + \
                                            str(disfluencyTagList) + \
                                            str(repair.to_string())
                                        if debug:
                                            print warning
                                        errors += 1
                                        if self.errorlog:
                                            self.errorlog.write(warning)
                                        raise Exception
                        if debug:
                            print tag
                        if tag:  # if there is a tag for this word
                            disfluency_tag += '<{} id="{}"/>'.format(
                                tag, repair.repairID)
                        if test_dodgy == my_test_dodgy and debug:
                            raw_input()

                    # in edit term but not in interregnum (forward looking
                    # disf)
                    if MD == True:
                        # TODO for now adding to every single tag
                        disfluency_tag += "<e/>"

                    # if we get this far, we're definitely adding this word,
                    # pos
                    wordList.append(word)
                    POSList.append(pos)
                    disfluencyTagList.append(disfluency_tag)
                    indexList.append(
                        (utt.utterance_index, utt.transcript_index, i))

                # end of word loop
                if (len(wordList) == 0):
                    # not adding empty strings- no need for detection purposes
                    if self.errorlog:
                        self.errorlog.write(
                            "no words in " + utt.swda_filename + ","
                            + str(utt.transcript_index) + "\n")
                    count += 1
                    continue
                if not len(wordList) == len(POSList) == len(disfluencyTagList):
                    warning = "WARNING uneven lengths between pos and \
                    word lists in  " + \
                        utt.swda_filename + str(utt.transcript_index) + "\n"
                    if debug:
                        print warning
                    warnings += 1
                    if self.errorlog:
                        self.errorlog.write(warning + "\n")
                    count += 1
                    raise Exception

                uttList.append([utt.swda_filename, utt.transcript_index,
                                utt.caller, utt.damsl_act_tag(), utt.conversation_no,
                                utt.utterance_index])
                overallWordsList.append(wordList)
                overallPOSList.append(POSList)
                overallIndexList.append(indexList)
                overallTagList.append(disfluencyTagList)
                count += 1

            # end of transcript loop
            # flush resolved repairs and find any unresolved ones
            while len(repairStack) > 0:
                for rs in range(0, len(repairStack)):
                    repair = repairStack[rs]
                    if repair.complete == True:  # should pop it off here
                        repairStack.pop(rs)
                        break
                    else:
                        find_repair_end_in_previous_utts(
                            repair, overallTagList, uttList)

        # end of corpus loop
        # concatenate utterances for disfluency detection with + DA tags
        # and those with crossing repair dependencies (unfinished repairs),
        # consistent with normal Switchboard task
        total = len(uttList)
        i = 0
        mergedUtts = 0
        while i < total:
            # check for + utterances and unfinished repairs
            if uttList[i][3] == '+' or \
                    (len(overallTagList[i]) > 0 and
                     (("<rp" in
                       overallTagList[i][0]
                       or "<i" in overallTagList[i][0])
                      or ("<rm " in overallTagList[i][0]))):
                if not uttList[i][3] == '+':
                    mergedUtts += 1
                found = False
                # backwards search for previous utt same speaker
                for o in range(i - 1, i - 20, -1):
                    if uttList[o][0] == uttList[i][0] and \
                            uttList[o][2] == uttList[i][2]:
                        uttList[o] = uttList[o] + uttList[i]
                        overallWordsList[o] = overallWordsList[
                            o] + overallWordsList[i]
                        overallPOSList[o] = overallPOSList[
                            o] + overallPOSList[i]
                        overallIndexList[o] = overallIndexList[
                            o] + overallIndexList[i]
                        overallTagList[o] = overallTagList[
                            o] + overallTagList[i]
                        del overallWordsList[i]
                        del uttList[i]
                        del overallPOSList[i]
                        del overallIndexList[i]
                        del overallTagList[i]
                        i = o  # go back to this one
                        total -= 1
                        found = True
                        #print("merged " + str(uttList[o]) + str(uttList[o]))
                        break
                if found == False:
                    warning = "ANNOTATION/MERGE ERROR: COULD NOT merge " + \
                        str(uttList[i])
                    print warning
                    errors += 1
                    if self.errorlog:
                        self.errorlog.write(warning + "\n")
                    raw_input()
                    i += 1
            else:
                i += 1
        print str(mergedUtts) + " utterances merged!"

        # Create the corpus as a string with the appropriate
        # utterance references
        # Check for illegal repair sequences and write warnings to error log
        # for checking
        dialogue_act = Counter()
        problem_repairs = 0
        easyread_corpus = ""
        corpus = ""
        for i in range(0, len(overallWordsList)):
            repairs_to_remove = []
            test_dodgy = uttList[i][4] + ":" + \
                str(uttList[i][2]) + ":" + str(uttList[i][1])
            for d in range(0, len(overallTagList[i])):
                if test_dodgy == my_test_dodgy and debug:
                    print "initial tag", d, overallTagList[i][d]
                    print '\ttags', overallTagList[i]
                # problem 0: interregnum but no edit term- need to adjust tags
                # up to the repair onset word, only if they're marked
                # independently like in swda
                if edit_terms_marked and ("<i" in overallTagList[i][d] and
                                          not "<e" in overallTagList[i][d]):
                    problem_interregs = re.findall(
                        "<i id\=\"[0-9]+\"\/>", overallTagList[i][d], re.S)
                    problem_interreg_IDs = []
                    for problem in problem_interregs:
                        repairID = problem[problem.find("=") + 2:-3]
                        problem_interreg_IDs.append(repairID)
                    remove_non_edit_interregna(
                        overallTagList[i],
                        overallWordsList[i],
                        problem_interreg_IDs)
                    warning = "ANNOTATION ERROR: Interregnum but no edit term.\
                     Shifting/removing interrengna. \n\t" + \
                        str(uttList[i]) + "\n\t" + str(overallTagList[i])
                    if debug:
                        print warning
                    errors += 1
                    if self.errorlog:
                        # deletes are special case where interregnum not always
                        # marked so needs to be added
                        self.errorlog.write(warning + "\n")
                # problem 1: interregnum or edit term overwrite key part of
                # repair
                if ("<i" in overallTagList[i][d] or "<e"
                    in overallTagList[i][d]) and \
                        ("<rps" in overallTagList[i][d] or
                         "<rpn" in overallTagList[i][d] or
                         "<rms" in overallTagList[i][d]):
                    if debug:
                        print "getting to problem (1)"
                    # Note the affected repairs from context all together
                    problem_rps = re.findall(
                        "<rps id\=\"[0-9]+\"\/>", overallTagList[i][d], re.S)
                    problem_rms = re.findall(
                        "<rms id\=\"[0-9]+\"\/>", overallTagList[i][d], re.S)
                    problem_rpns = re.findall("<rpnrep id\=\"[0-9]+\"\/>",
                                              overallTagList[i][d], re.S) +\
                        re.findall(
                            "<rpnsub id\=\"[0-9]+\"\/>", overallTagList[i][d],
                            re.S)
                    problem_rpns_del = re.findall(
                        "<rpndel id\=\"[0-9]+\"\/>", overallTagList[i][d],
                        re.S)
                    if problem_rpns_del:  # deletes a special case
                        # delete these rpn repair start/end tags
                        for r in problem_rpns_del:
                            overallTagList[i][d] = overallTagList[i][d].\
                                replace(
                                r, "").replace(r.replace("rpndel", "rps"), "")
                        # shift all repair deletes along to next non-edit term
                        resolved = find_delete_interregna_and_repair_onsets(
                            overallTagList[i], problem_rpns_del, d)

                        if len(resolved) > 0:
                            for r in resolved:
                                problem_rpns_del.remove(r)
                                problem_rps.remove(r.replace("rpndel", "rps"))
                            warning = "ANNOTATION WARNING: No interregnum \
                            marked for delete(s). Shifting delete annotations\
                             over an interregnum. \n\t" + str(
                                uttList[i])
                            warnings += 1
                            if debug:
                                print warning
                            if self.errorlog:
                                # deletes are special case where interregnum
                                # not always marked so needs to be added
                                self.errorlog.write(warning + "\n")

                    if problem_rpns:
                        # Try to shift the repair end backwards
                        if debug:
                            print "problem rpns!"
                        unresolved = []
                        for r in problem_rpns:
                            if r.replace("rpnsub", "rps") in problem_rps or\
                                    r.replace("rpnrep", "rps") in problem_rps:
                                # don't deal with one word repairs, that will
                                # be dealt with by the repair onset solution
                                # below
                                continue
                            unresolved.append(r)

                        # first get rid of offending ones
                        for r in problem_rpns:
                            overallTagList[i][d] = overallTagList[
                                i][d].replace(r, "")
                        resolved = find_repair_ends_and_reclassify(
                            unresolved, overallTagList[i], overallWordsList[i],
                            d)
                        if len(resolved) > 0:
                            for r in resolved:
                                problem_rpns.remove(r)
                            warning = "ANNOTATION WARNING: \
                            Shifting repair end back left of \
                            an edit term. \n\t" + \
                                str(uttList[i])
                            if debug:
                                print warning
                            warnings += 1
                            if self.errorlog:
                                # deletes are special case where interregnum
                                # not always marked so needs to be added
                                self.errorlog.write(warning + "\n")

                    # if possible to shift rms to a non-edit term word before
                    # gettint to the repair onset word
                    if problem_rms:
                        # remove these problem rms tags from the tag
                        resolved = []  # rms which can be shifted successfully
                        unresolved = []
                        for r in problem_rms:
                            overallTagList[i][d] = overallTagList[
                                i][d].replace(r, "")

                        for fd in range(d, len(overallTagList[i])):
                            for r in problem_rms:
                                if r in unresolved or r in resolved:
                                    continue
                                repair_start_test = r.replace("rms", "rps")
                                # once past the repair start, too late
                                if repair_start_test in overallTagList[i][fd]:
                                    unresolved.append(r)
                                    continue
                                # legit reparandum start
                                if not "<e" in overallTagList[i][fd] and \
                                        not "<i" in overallTagList[i][fd]:
                                    overallTagList[i][fd] = overallTagList[i][
                                        fd].replace(
                                        r.replace("rms",
                                                  "rm"), "") + r
                                    resolved.append(r)
                        if len(resolved) > 0:
                            for r in resolved:
                                problem_rms.remove(r)
                            warning = "ANNOTATION WARNING: Shifting reparandum\
                             onset right of an edit term. \n\t" + \
                                str(uttList[i])
                            if debug:
                                print warning
                            warnings += 1
                            if self.errorlog:
                                # deletes are special case where interregnum
                                # not always marked so needs to be added
                                self.errorlog.write(warning + "\n")

                    if problem_rps:
                        # Try to shift the repair forward
                        # first delete the bad repair onset(s)
                        for r in problem_rps:
                            overallTagList[i][d] = overallTagList[
                                i][d].replace(r, "")
                        # find new repair onset/interregna all repair onsets
                        # along to next non-edit term and mark the interrengum
                        # if possible
                        if debug:
                            print "removing rps", overallTagList[i][d]
                        resolved = find_interregna_and_repair_onsets(
                            overallTagList[i], problem_rps, d,
                            overallWordsList[i])

                        if len(resolved) > 0:
                            for r in resolved:
                                problem_rps.remove(r)
                            warning = "ANNOTATION WARNING: \
                            No interregnum marked for repeat/subs. \
                            Shifting annotations over an interregnum. \n\t" + \
                                str(uttList[i])
                            warnings += 1
                            if debug:
                                print warning
                            if self.errorlog:
                                # deletes are special case where interregnum
                                # not always marked so needs to be added
                                self.errorlog.write(warning + "\n")

                    # the remaining ones to problematic to save, discard and
                    # warn
                    if problem_rps or problem_rms or problem_rpns \
                            or problem_rpns_del:
                        warning = "ANNOTATION ERROR: Edit term or \
                        interregnum overwriting key repair element. \
                        Removing repair. \n\t" + \
                            str(uttList[i])
                        if debug:
                            print warning
                        errors += 1
                        if self.errorlog:
                            # deletes are special case where interregnum not
                            # always marked so needs to be added
                            self.errorlog.write(warning + "\n")
                        repairs_to_remove.extend(problem_rps)
                        repairs_to_remove.extend(problem_rms)
                        repairs_to_remove.extend(problem_rpns)
                        repairs_to_remove.extend(problem_rpns_del)

                if ("<i" in overallTagList[i][d] or
                    "<e" in overallTagList[i][d]) and \
                    ("<rp " in overallTagList[i][d] or
                     "<rm " in overallTagList[i][d]):
                    # problem 2: interregnum or edit term overwriting non-key
                    # part of repair
                    warning = "ANNOTATION WARNING: Interregnum or \
                    edit term over-writing mid-reparandum or \
                    mid-repair element. \n\t" + \
                        str(uttList[i])
                    if debug:
                        print warning
                    warnings += 1
                    if self.errorlog:
                        self.errorlog.write(warning + "\n")
                    edit = ""
                    if "<e" in overallTagList[i][d]:
                        edit = "<e/>"
                    if debug:
                        print "refinall interreg",
                        re.findall("<i id\=\"[0-9]+\"\/>",
                                   overallTagList[i][d], re.S)
                    overallTagList[i][
                        d] = edit + "".join(re.findall("<i id\=\"[0-9]+\"\/>",
                                                       overallTagList[i][d],
                                                       re.S))
                    if debug:
                        print "tags after removal", overallTagList[i][d]

                else:
                    # problem 3: embedded repairs (two or more repairs with
                    # same onset word position)
                    repair_onsets = re.findall(
                        "<rps id\=\"[0-9]+\"\/>", overallTagList[i][d], re.S)
                    if len(repair_onsets) > 1:  # there's an embedded repair
                        warning = "ANNOTATION ERROR: Embedded repairs. \n\t" +\
                            str(uttList[i])
                        errors += 1
                        if debug:
                            print warning
                        if self.errorlog:
                            self.errorlog.write(warning + "\n")
                        embedded_repairs_to_remove = []
                        repair_types = {'sub': [], 'del': [], 'rep': []}
                        # backwards search from end of utterance
                        for b in range(len(overallTagList[i]) - 1, -1, -1):
                            # get repair ends of the suspicious repairs first
                            for key in repair_types.keys():
                                for rps in repair_onsets:
                                    rpn = rps.replace("rps", "rpn" + key)
                                    if rpn in overallTagList[i][b]:
                                        repair_types[key].append(rps)
                            # get the reparandum onsets of the suspicious
                            # repairs that start on this word
                            reparandum_onsets = []
                            for rps in repair_onsets:
                                rms = rps.replace("p", "m")
                                # reparandum start for repair
                                if rms in overallTagList[i][b] and not rps in\
                                        reparandum_onsets:
                                    reparandum_onsets.append(rps)
                            # if more than one suspicious repair reparandum
                            # onset for this word
                            # remove by type
                            while len(reparandum_onsets) > 1:
                                # remove deletes, then subs, then reps
                                if debug:
                                    print reparandum_onsets
                                if len(repair_types['del']) > 0:
                                    for index in range(
                                            0,
                                            len(repair_types['del'])):
                                        test = repair_types['del'][index]
                                        if test in reparandum_onsets:
                                            repair_types['del'].remove(test)
                                            reparandum_onsets.remove(test)
                                            embedded_repairs_to_remove.append(
                                                test)
                                            break
                                elif len(repair_types['sub']) > 0:
                                    for index in range(
                                            0,
                                            len(repair_types['sub'])):
                                        test = repair_types['sub'][index]
                                        if test in reparandum_onsets:
                                            repair_types['sub'].remove(test)
                                            reparandum_onsets.remove(test)
                                            embedded_repairs_to_remove.append(
                                                test)
                                            break
                                elif len(repair_types['rep']) > 0:
                                    for index in range(
                                            0,
                                            len(repair_types['rep'])):
                                        test = repair_types['rep'][index]
                                        if test in reparandum_onsets:
                                            repair_types['rep'].remove(test)
                                            reparandum_onsets.remove(test)
                                            embedded_repairs_to_remove.append(
                                                test)
                                            break

                            # add the last remaining suspicious reparandum
                            # onset for removal if possible
                            if len(repair_onsets) - \
                                    len(embedded_repairs_to_remove) > 1:
                                embedded_repairs_to_remove.extend(
                                    reparandum_onsets)
                            # if down to the one valid repair then end loop
                            if len(repair_onsets) - \
                                    len(embedded_repairs_to_remove) == 1:
                                break

                        repairs_to_remove.extend(embedded_repairs_to_remove)

            problem_repairs += len(repairs_to_remove)
            showchange = False
            if debug and len(repairs_to_remove) > 0:
                print easy_read_disf_format(overallWordsList[i],
                                            overallTagList[i])
                showchange = True  # Todo which it
            problem_IDs = []
            for problem in repairs_to_remove:
                repairID = problem[problem.find("=") + 2:-3]
                if not repairID in problem_IDs:
                    problem_IDs.append(repairID)
            overallTagList[i] = remove_repairs(overallTagList[i], problem_IDs)
            if showchange and debug:
                print problem_IDs
                print easy_read_disf_format(overallWordsList[i],
                                            overallTagList[i])

            if test_dodgy == my_test_dodgy and debug:
                raw_input()

            # now check for possibly missed filled pause edit terms or
            # interregna
            for df in range(0, len(overallTagList[i])):
                my_disfluency_tag = overallTagList[i][df]
                if overallWordsList[i][df] in filled_pauses and \
                        not ("<e" in my_disfluency_tag or
                             "<i" in my_disfluency_tag):
                    warning = "ANNOTATION WARNING: possibly missed \
                    filled pause edit term!\n\t" + \
                        str(uttList[i]) + str(overallWordsList[i]) + \
                        str(overallTagList[i])
                    if debug:
                        print warning
                    warnings += 1
                    if self.errorlog:
                        self.errorlog.write(warning + "\n")
                    continue

            # if there are 'relaxed' per line repair annotations, add them here
            # to the tags
            # if repaired utterances add their annotations here
            if self.dRelaxedRepairs:
                # conversation_no and transcript_index
                relaxedRef = str(uttList[i][4]) + ":" + str(uttList[i][1])
                # mutable list- adds tags in place
                annotate_relaxed_repairs(
                    overallTagList[i], self.dRelaxedRepairs[relaxedRef])

            if debug:
                print overallWordsList[i], overallTagList[i]
            # No partial words allowed, remove them
            if not partial:
                removal = remove_partial_words(
                    deepcopy(overallTagList[i]), deepcopy(overallWordsList[i]),
                    deepcopy(overallPOSList[i]), deepcopy(overallIndexList[i]))
                overallTagList[i] = removal[0]
                overallWordsList[i] = removal[1]
                overallPOSList[i] = removal[2]
                overallIndexList[i] = removal[3]
                # if debug:
                print 'after partials removed\n',
                print str(overallWordsList[i]) + "\n", str(overallTagList[i]), overallIndexList[i]
                if len(overallTagList[i]) == 0:
                    print "ERROR empty list"
                    continue
            # normalise tag IDs and check them
            verify_disfluency_tags(overallTagList[i], normalize_ID=True)

            # now get the final strings for the corpus string for writing to
            # file
            extra = ""
            # TODO deal with continuation utts or not?
            # if len(uttList[i])>6:
            #    print uttList[i]
            #    print float(len(uttList[i]))/ 6.
            #    extra = int((float(len(uttList[i]))/ 6.) - 1.) * "+"
            #    print extra
            d_act = uttList[i][3]
            # nb adding extra info into the uttref of dialogue acts
            if self.d_acts:
                extra = ":" + d_act
            # print d_act
            dialogue_act[d_act] += 1
            uttref = uttList[i][4] + \
                ":" + str(uttList[i][2]) + ":" \
                + str(uttList[i][1]) + extra
            if d_act == "o/aa":
                print "dialogue act bad"
                print uttref
                print uttList[i]
                raw_input()
            wordstring = easy_read_disf_format(
                overallWordsList[i], overallTagList[i])
            posstring = easy_read_disf_format(
                overallPOSList[i], overallTagList[i])
            indexstring = ""
            indices = []
            for myInd in overallIndexList[i]:
                indices.append(
                    "({}:{}:{})".format(myInd[0], myInd[1], myInd[2]))
            indexstring = " ".join(indices)

            # Can do easyread to the corpus string and/or more kosher xml tags
            # like the ATIS shared challenge
            easyread_format = uttref + "," + wordstring + \
                "\nPOS," + posstring + "\n" + "REF," + indexstring
            corpus_format = detection_corpus_format(
                uttref, deepcopy(overallWordsList[i]),
                deepcopy(overallPOSList[i]),
                deepcopy(overallTagList[i]), deepcopy(indices))
            # check one can be derived from the other
            assert easyread_format == easy_read_format_from_detection_corpus(
                corpus_format), easyread_format + "VS\n" + \
                easy_read_format_from_detection_corpus(corpus_format)
            assert corpus_format == detection_corpus_format_from_easy_read(
                easyread_format), corpus_format + "VS\n" + \
                detection_corpus_format_from_easy_read(easyread_format)
            # add to corpus, new line separated
            corpus += corpus_format + "\n"
            easyread_corpus += easyread_format + "\n"
            # counting
            totalRepairs += len(re.findall("<rps", wordstring, re.S))
            wordCount += len(wordstring.split())

        # write corpus string to file
        if partial:
            filepartial = "_partial"  # suffix for file name
        else:
            filepartial = ""
        if writeFile:
            disffile = open(filename + filepartial + "_data.csv", "w")
            disffile.write(corpus)
            disffile.close()

        print>>self.errorlog, "Problematic repairs removed = " + \
            str(problem_repairs)
        print "Problematic repairs removed = " + str(problem_repairs)
        print>>self.errorlog, "Disfluency corpus constructed with\
         {} Errors and {} Warnings".format(
            errors, warnings)
        print "Disfluency corpus constructed with \
        {} Errors and {} Warnings".format(errors, warnings)
        print "Disfluency detection corpus complete"
        print "number trans = " + str(numberTrans)
        print "number utts = " + str(len(uttList))
        print "number words = " + str(wordCount)
        print "number repairs = " + str(totalRepairs)
        for k, v in sorted(dialogue_act.items(), key=lambda x: x[1],
                           reverse=True):
            print k, v
        return easyread_corpus  # Always return the corpus as a string

    def write_clean_corpus(self, testcorpus, targetfilename, debug=False):
        """Write a file cleaned of reparanda and edit terms from a test corpus.

        Keyword Arguments:
        testcorpus -- a string separated by newline markers which
        has all the disfluency markup.
        targetfilename -- a string giving the location of the cleaned corpus.
        """
        print "Writing clean corpus..."
        clean_corpus = open(targetfilename + "_clean.text", "w")
        for line in testcorpus.split("\n"):
            # print line
            if line == "":
                continue
            split = line.split(",")
            # no need to write the indices to source data
            if split[0] == "REF":
                continue
            elif split[0] == "POS":
                pos = split[1]
            else:
                uttref = split[0]
                text = split[1]
                continue

            words = strip_disf_tags_from_easy_read(text)
            pos_tags = strip_disf_tags_from_easy_read(pos)
            disfluencies = disf_tags_from_easy_read(text)
            clean_word_string = ""
            clean_pos_string = ""
            for i in range(0, len(disfluencies)):
                if "<e" in disfluencies[i] or "<rm" in disfluencies[i] \
                        or "<i" in disfluencies[i]:
                    continue
                clean_word_string += words[i] + " "
                clean_pos_string += pos_tags[i] + " "
            clean_word_string = clean_word_string.strip()
            clean_pos_string = clean_pos_string.strip()
            if clean_word_string == "":
                continue
            clean_corpus.write(
                uttref + "," + clean_word_string + "\nPOS," +
                clean_pos_string + "\n")
        clean_corpus.close()
        print "done"
        return

    def write_edit_term_corpus(self, testcorpus, targetfilename, debug=False):
        """Write a file cleaned of reparanda and edit terms from a test corpus.

        Keyword Arguments:
        testcorpus -- a string separated by newline markers which
        has all the disfluency markup.
        targetfilename -- a string giving the location of
        the edit term filled corpus.
        """
        print "Writing edit term corpus..."
        edit_term_corpus = open(targetfilename + "_edit.text", "w")
        for line in testcorpus.split("\n"):
            if line == "":
                continue
            split = line.split(",")
            # no need to write the indices to source data
            if split[0] == "REF":
                continue
            elif split[0] == "POS":
                pos = split[1]
            else:
                uttref = split[0]
                text = split[1]
                continue
            editterm_examples = get_edit_terms_from_easy_read(text, pos)
            if editterm_examples != []:
                for my_editterm, my_poseditterm in editterm_examples:
                    edit_term_corpus.write(
                        uttref + "," + my_editterm + "\nPOS," +
                        my_poseditterm + "\n")
        edit_term_corpus.close()
        print "done"
        return

#     def write_tree_corpus(self, testcorpus, targetfilename, debug=False):
#         #TODO finish and test
#         reffile = open(targetfilename+"TreeCorpus.text","w")
#         overallCount = 0
#         stringlist = []
#         uttlist = []
#         overallPOSList = []
#         overallIndexList = []
#         dodgy = ["NNP","NNPS","CD","LS","SYM","FW"]
#         totalRepairs = 0
#         thirdPos = 0
#         firstPos = 0
#         typedict = defaultdict(list)
#         printdict = defaultdict(list)
#
#         self.editingTermList = [[]]
#         # preprocessing step to get all the editing terms seen
#         #in whole corpus, list of lists
#         #what do we do for held out data again..,
#         # for realistic test should see how we do with the edit dict.
#         numberTrans = 0
#         for trans in self.corpus.iter_transcripts():
#             repairStack = [] #can be both speakers/callers or just one
#             count=0
#             transnumber = int(trans.swda_filename[19:23])
#             #conversationNumber = int(trans.conversation_no)
#             if transnumber < 1: continue #from
#             #if ranges and not trans.has_trees():
#             #    continue
#
#
#             if transnumber > 1210: break
#             #up to test files 1210, none beyond this
#
#             numberTrans+=1
#             while count < len(trans.utterances):
#                 utt = trans.utterances[count]
#                 mytreemap = None
#                 myPOSmap = None
#                 treepaths = None #will be a defaultdict
#                 if trans.has_trees():
#                     mytreemap = self.__treemaplist__.get_treemap(trans,utt)
#                     if mytreemap:
#                         mypos = mytreemap.get_POS(trans,utt)
#                         try:
#                             treepaths = mytreemap.get_path_lengths(
#                                                 trans,utt.transcript_index)
#                         except:
#                             print "treepath problem"
#                             print sys.exc_info()
#                     else:
#                         count+=1
#                         continue
#                 else: # for POS files
#                     myPOSmap = self.__POSmaplist__.get_POSmap(trans,utt)
#                     if myPOSmap:
#                         mypos = myPOSmap.get_POS(utt)
#                     else:
#                         count+=1
#                         continue
#                 #TODO get corresponding repair line from the test corpus
#         return

    def disfluency_type_corpus_analysis(self, testcorpus, targetfilename):
        """Take a test corpus and compute distributions over
        the types of disfluencies.
        For repairs, the types we're looking at are the structures
        from the minimal edit distance alignment of repair and reparandum.
        For edit terms/interregna this is the probability of words
        being edit terms at all, and the probability of
        them being interregna.

        Keyword Arguments:
        testcorpus -- a string separated by newline markers which
        has all the disfluency markup.
        targetfilename -- a string giving the location of
        the edit term filled corpus.
        """
        # TODO finish and test
        typedict = defaultdict(list)
        printdict = defaultdict(list)
        editingTermList = [[]]
        for line in testcorpus.split("\n"):
            if line == "":
                continue
            split = line.split(",")
            ref = split[0]
            text = split[1]
            if ref == "REF":
                continue
            disfluencies = disf_tags_from_easy_read(text)
            for i in range(0, len(disfluencies)):
                if "<rpn" in disfluencies[i]:
                    repair = None  # TODO place holder
                    # TODO get the general class rep/sub/delete from the words
                    simpleclass = None
                    try:
                        if not repair.reparandumWords == []:
                            complexclass = classify_repair(
                                repair.reparandumWords,
                                repair.repairWords,
                                repair.continuationWords
                            )
                            # ONLY ADD ONE EXAMPLE, the top one..
                            if not complexclass in typedict[simpleclass]:
                                output = graph_viz_repair(
                                    complexclass, repair.reparandumWords,
                                    repair.repairWords,
                                    repair.continuationWords)
                                if not output in printdict[simpleclass]:
                                    printdict[simpleclass].extend(output, 1)
                                typedict[simpleclass].append(
                                    [complexclass, output, 1])
                            else:
                                for mytype in typedict[simpleclass]:
                                    if mytype[0] == complexclass:
                                        # allows us to get the count for each
                                        # complex type
                                        mytype[2] += 1

                    except BaseException:
                        print sys.exc_info()
                        print repair.reparandumWords
                        print repair.repairWords
                        print repair.continuationWords
                        raw_input("third one")
        # Do the interregnum and edit term analysis
        interregDict = defaultdict(int)
        for edit in editingTermList:
            if edit == []:
                continue
            editstring = ""
            for string in edit:
                editstring += string + " "
            editstring = editstring[:-1]
            interregDict[editstring] += 1
        # now turn it into a dict list for corpus analysis
        # this will be used in below method, dict of [incleanCorpus,FLD,Repair]
        editingTermList = defaultdict(list)

        # creates
        for w in sorted(interregDict, key=interregDict.get, reverse=True):
            print w, interregDict[w]
            editingTermList[w] = [0, 0, 0]
            # interregFile.write(w+","+str(interregDict[w])+"\n")
        # interregFile.close()

        typefile = open(targetfilename + "RepairTypeDistributions.text", "w")
        print "writing to typefile" + typefile.name
        for key in printdict.keys():
            print str(key) + str(len(typedict[key]))
            typefile.write("\n\n\n\n\n" + str(key) + "MAINTYPE:" +
                           str(len(printdict[key])) +
                           "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            # writes the most popular mappings-- should do this is order of
            # occurence?
            for repairtype in printdict[key]:
                typefile.write(repairtype + "\n")
        # for key in typedict.keys():
        #    print key
        #    print
        typefile.close()
        return

    def make_corpus(self, target_folder=None, corpus=None, range_files=None,
                    edit_terms_marked=True, mode=None, debug=False):
        """Write a disfluency related corpus according to the
        mode required."""

        self.dRepairs = defaultdict(list)
        # all the edit terms ranges, may be able to get round this easier below
        self.dEditTerms = defaultdict(list)
        self.dRelaxedRepairs = None

        # get the repair annotations
        if self.annotationFiles:
            self.read_in_self_repair_files()
        if self.annotationFilesRelaxed:
            self.read_in_relaxed_self_repair_files

        self.ranges = []
        # get the ranges of the source corpus to extract data from
        if range_files:
            for range_file_name in range_files:
                rangeFile = open(range_file_name, "r")
                for line in rangeFile:
                    a = line.strip("\n")
                    self.ranges.append(a)  # conversation_no
                    # print a
                rangeFile.close()
            print "files in ranges = " + str(len(self.ranges))
        else:
            self.ranges = None
        # write the corpus for the accepted mode
        if mode in ["both", "clean", "disfluency"]:
            if mode == "disfluency":
                write_edit = False
            else:
                write_edit = True
            if mode == "clean":
                save_test = False
            else:
                save_test = True

            test_corpus = self.write_test_corpus(target_folder + os.sep +
                                                 corpus,
                                                 ranges=self.ranges,
                                                 partial=args.partialWords,
                                                 writeFile=save_test,
                                                 edit_terms_marked=edit_terms_marked,
                                                 debug=debug)
            if mode in ["clean", "both"]:
                self.write_clean_corpus(
                    test_corpus, target_folder + os.sep + corpus, debug=debug)
            if write_edit:
                self.write_edit_term_corpus(
                    test_corpus, target_folder + os.sep + corpus, debug=debug)
        if mode == "tree":
            self.write_tree_corpus(target_folder + os.sep + corpus + os.sep +
                                   "TREEPATHS" + corpus, ranges=self.ranges,
                                   partial=self.partial_words)


if __name__ == '__main__':
    # parse command line parameters
    # Optional arguments:
    # -i string, path of source data (in swda style)
    # -t string, target path of folder for the preprocessed data
    # -f string, path of file with the division of files to be turned into
    # a corpus
    # -a string, path to disfluency annotations
    # -l string, Location of where to write a clean language\
    # model files out of this corpus
    # -pos boolean, Whether to write a word2pos mapping folder
    # in the sister directory to the corpusLocation, else assume it is there
    # -p boolean, whether to include partial words or not
    # -d boolean, include dialogue act tags in the info
    parser = argparse.ArgumentParser(description='Feature extraction for\
    disfluency and other tagging tasks from raw data.')
    parser.add_argument('-i', action='store', dest='corpusLocation',
                        default='../data/raw_data/swda',
                        help='location of the corpus folder')
    parser.add_argument('-t', action='store', dest="targetDir",
                        default='../data/disfluency_detection/switchboard')
    parser.add_argument('-f', action='store', dest='divisionFile',
                        default='../data/disfluency_detection/\
                        swda_divisions_disfluency_detection/\
                        SWDisfTrain_ranges.text',
                        help='location of the file listing the \
                        files used in the corpus')
    parser.add_argument('-a', action='store', dest='annotationFolder',
                        default='../data/disfluency_detection/\
                        swda_disfluency_annotatations',
                        help='location of the disfluency annotation csv file')
    parser.add_argument('-lm', action='store', dest='cleanModelDir',
                        default=None,
                        help='Location of where to write a clean language\
                            model files out of this corpus.')
    parser.add_argument('-pos', action='store_true', dest='writePOSdir',
                        default=False,
                        help='Whether to write a word2pos mapping\
                        in the sister directory to the corpusLocation.')
    parser.add_argument('-d', action='store_true', dest='dialogueActs',
                        default=False,
                        help='Whether to annotate with dialogue acts.')
    parser.add_argument('-p', action='store_true', dest='partialWords',
                        default=False,
                        help='Whether to include partial words or not.')
    args = parser.parse_args()

    # Example from new data with only relaxed utterances
    # available and no pos maps

    # Example from switchboard data where tree/pos maps already created
    treeposmapdir = args.corpusLocation + '/../swda_tree_pos_maps'
    if args.writePOSdir:
        try:
            os.mkdir(treeposmapdir)  # If POSmap doesn't exist yet, make one
        except OSError:
            print "already made POS map dir", treeposmapdir
        print "writing a word -> pos mapping directory"
        TreeMapWriter(args.corpusLocation,
                      target_folder_path=treeposmapdir,
                      ranges=None,
                      errorLog="POSmap_error_log.text")
        POSMapWriter(args.corpusLocation,
                     target_folder_path=treeposmapdir,
                     ranges=None,
                     errorLog="POSmap_error_log.text")
    if not args.annotationFolder:
        print "WARNING NO DISFLUENCY ANNOTATION FOLDER! Quitting"
        quit()
    annotation_files = [filename for filename in
                        iglob(os.path.join(args.annotationFolder,
                                           "*.csv.text"))]
    n = DisfluencyCorpusCreator(args.corpusLocation,
                                metadata_path="swda-metadata.csv",
                                mapdir_path=treeposmapdir,
                                partial=args.partialWords,
                                annotation_files=annotation_files,
                                error_log="swbd_error_log.text")

    corpusName = args.divisionFile[args.divisionFile.rfind("/") + 1:].\
        replace("_ranges.text", "")
    # (1) Make disfluency detection annotated files
    n.make_corpus(target_folder=args.targetDir,
                  corpus=corpusName,
                  range_files=[args.divisionFile],
                  mode="disfluency",
                  debug=False)

    # (2) Make clean and edit term files
    if args.cleanModelDir:
        n.make_corpus(target_folder=args.cleanModelDir,
                      corpus=corpusName,
                      range_files=[args.divisionFile],
                      mode="clean",
                      debug=False)
