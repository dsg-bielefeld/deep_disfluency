# -*- coding: utf-8 -*-
"""A script to map the SWDA disfluency transcriptions to the
Mississippi word aligned transcripts.
Then add to an enriched gold standard file with accurate word
timing info.

It also adds the laughter tags (speech laughs and bouts) and the dialogue
act tags and BIES for dialogue act segmentation to the representation.
i.e. all the possible tags
"""

import argparse
from copy import deepcopy
#import sys
# ssys.setrecursionlimit(10000) #NB shouldn't be needed

#from mumodo.mumodoIO import open_intervalframe_from_textgrid
##from mumodo.mumodoIO import save_intervalframe_to_textgrid

from word_alignment import align
from feature_utils import load_data_from_disfluency_corpus_file
from feature_utils import sort_into_dialogue_speakers
from functools import reduce


def get_best_word_alignment(ms_words, swda_words, ms_start_times,
                            ms_end_times, swda_mappings=None, MSUttIndex=None,
                            debug=False):
    """Takes a list of words from the ms aligned files and a list \
    of utterances from the SWDA files, aligns them and outputs
    a list of the same length as the ms_words input but with the \
    best mapping from ms_words -> SWDA files
    """
    if debug:
        if any(["<laughter>" in x for x in ms_words]):
            print "laughter before mapping"
            print ms_words, swda_words
            # raw_input()
    alignment = align(ms_words, swda_words)  # gets best alignment
    if debug:
        if any(["<laughter>" in x for x in ms_words]):
            print "laughter after alignment"
            print ms_words, swda_words
            print alignment
            print swda_mappings
            # raw_input()
    #final_alignment = [[word] for word in ms_words]

    # print final_alignment
    assert ms_start_times != []
    assert ms_end_times != []
    if alignment == []:
        return [], []

    # print "MS swda words"
    # for a,b,c in zip(ms_words,ms_start_times,ms_end_times):
    #    print a,b,c
    # print "SWDA words"
    # print "*" * 20
    # for a,b,c in zip(swda_words,ms_start_times,ms_end_times):
    #    print a,b,c

    # print alignment

    # output a list of alignments in the same format as MS files ->
    # for lists of
    # mswords + the alignment relation
    # to swdawords, swdawords, starttimes, stoptimes,
    final_swda_words = []
    final_ms_words_mapping = []  # words and the mappings
    final_ms_start_times = []
    final_ms_end_times = []
    for a in range(0, len(alignment)):
        if alignment[a][2] == "DEL":
            final_ms_words_mapping.append((ms_words[alignment[a][0]], "<DEL>"))
            final_swda_words.append("")  # nothing for the swda word
            # normal start and end time
            final_ms_start_times.append(ms_start_times[alignment[a][0]])
            final_ms_end_times.append(ms_end_times[alignment[a][0]])
        elif alignment[a][2] == "INS":
            final_ms_words_mapping.append(("", "<INS>"))
            final_swda_words.append(swda_words[alignment[a][1]])
            if len(final_ms_end_times) > 0:
                # keep the same as previous end
                final_ms_start_times.append(final_ms_end_times[-1])
                # no length as words are inserted
                final_ms_end_times.append(final_ms_end_times[-1])
            else:
                # just keep inserting words with 0 duration
                final_ms_start_times.append(ms_start_times[0])
                final_ms_end_times.append(ms_start_times[0])
        else:  # assume normal sub/link
            final_ms_words_mapping.append((ms_words[alignment[a][0]], ""))
            final_swda_words.append(swda_words[alignment[a][1]])
            final_ms_start_times.append(ms_start_times[alignment[a][0]])
            final_ms_end_times.append(ms_end_times[alignment[a][0]])

    return (final_ms_words_mapping, final_swda_words, final_ms_start_times,
            final_ms_end_times, alignment)


def timings_for_MS_SWDAwords_from_MSwords(current_MSwords,
                                          current_MS_SWDAwords,
                                          current_MS_start_times,
                                          current_MS_end_times,
                                          data=None):
    # use the alignments in current_MSwords to get to current_MS_SWDAwords,
    # return the (estimated if needs be) start and end times of the SWDA words,
    # and the SWDAwords themselves to be used
    #inherit_last_start_time = False
    final_SWDA_words = []
    final_start_times = []
    final_end_times = []
    final_delete = False
    final_insert = None  # can be instantiated

    for i in range(0, len(current_MSwords)):
        # adjust end/start times based on the relation
        align = current_MSwords[i][1].strip()

        if align == "<DEL>":  # delete the SWDA word
            if i < len(current_MSwords) - 1:
                # print "DEL case 1"
                # shift the start time to the next word, spreading the length
                current_MS_start_times[i + 1] = current_MS_start_times[i]
                continue
            else:
                # print "DEL case 2!"
                # need to carry over the timing to the next utterance?,
                # or just ignore
                final_delete = True
        elif align == "<INS>":
            # get all the inserted words that may be in this bit
            if i > 0 and current_MSwords[i - 1][1].strip() == "<INS>":
                # if a continuation of an INS will have been dealt with
                continue
            inserted_words = []
            j = i  # else search for all next INS
            while current_MSwords[j][1].strip() == "<INS>":

                if "</laughter>" in current_MSwords[j][0] and \
                        not "<laughter" in current_MS_SWDAwords[j] and \
                        not clean_word(current_MS_SWDAwords[j]) == "":
                    current_MS_SWDAwords[j] = "<laughter>" +\
                        current_MS_SWDAwords[j] + "</laughter>"
                inserted_words.append(current_MS_SWDAwords[j])
                j += 1
                if j == len(current_MSwords):
                    j = j - 1
                    break
            # we have inserted words, give them start times of the previous word
            # (or next one after the INS tags if its the beginning of the utt)
            if len(final_start_times) > 0:
                # print "INS case 1"
                shared_length = final_end_times[-1] - final_start_times[-1]
                number_of_letters = len(reduce(lambda x, y: x + y,
                                               inserted_words + [final_SWDA_words[-1]]))
                # approx to the length of each phoneme
                av_length = shared_length / float(number_of_letters)
                current_time = final_start_times[-1] + (av_length *
                                                        len(final_SWDA_words[-1]))
                # adjust the end time of the previous word
                final_end_times[-1] = current_time

                for ins_word in inserted_words:
                    final_SWDA_words.append(ins_word)
                    final_start_times.append(current_time)
                    final_time = current_time + (av_length * len(ins_word))
                    final_end_times.append(final_time)
                    current_time = final_time

            elif i < len(current_MSwords) - 1:  # use the next word timing
                # print "INS case 2"
                # print current_MS_SWDAwords
                # print current_MSwords
                # print current_MS_end_times
                # print current_MS_start_times
                # print i
                j = i + 1
                while j < len(current_MS_end_times) and \
                        current_MS_end_times[j] == current_MS_start_times[j] or \
                        current_MSwords[j][1].strip() == "<DEL>":
                    j += 1
                    if j == len(current_MS_end_times):
                        final_insert = inserted_words
                        j = j - 1
                        break
                if j == len(current_MS_end_times):
                    j = j - 1
                if final_insert:
                    break  # TODO how do we carry over the inserted words??
                # we have some length for the next word if we get here
                shared_length = current_MS_end_times[j] - \
                    current_MS_start_times[j]
                number_of_letters = len(reduce(lambda x, y: x + y,
                                               inserted_words + [current_MSwords[j][0]]))
                av_length = shared_length / float(number_of_letters)

                current_time = current_MS_start_times[j]

                for ins_word in inserted_words:
                    final_SWDA_words.append(ins_word)
                    final_start_times.append(current_time)
                    final_time = current_time + (av_length * len(ins_word))
                    final_end_times.append(final_time)
                    current_time = final_time
                current_MS_start_times[j] = current_time
                # adjust the word whose time was shared
                if current_MS_end_times[j] - current_MS_start_times[j] > 0.0:
                    print "TIMING ERROR", data, "prev time", current_MS_end_times[j]
            else:
                # print "INS case 3"
                final_insert = inserted_words
                # TODO problematic ones, may have left over words with no
                # mappings- need to carry them over to next
                break

        elif align == "<CONT>":
            # ambig tag, can mean either MS is continuation or SWDA is,
            # need to account for both
            # will always deal with a new stream of CONT tags
            if i > 0 and current_MSwords[i - 1][1].strip() == "<CONT>":
                # if a continuation follows will have been dealt with
                continue
            cont_swda = []
            cont_ms = []
            j = i  # else search for all next CONT
            end_time = current_MS_end_times[i]
            # print current_MSwords[j:]
            while current_MSwords[j][1].strip() == "<CONT>":
                if current_MSwords[j][0] != "":
                    cont_ms.append(current_MSwords[j][0])
                if current_MS_SWDAwords[j] != "":
                    cont_swda.append(current_MS_SWDAwords[j])
                end_time = current_MS_end_times[j]
                j += 1
                if j == len(current_MSwords):
                    j = j - 1
                    break
            # print cont_ms
            # print cont_swda
            assert not (len(cont_swda) == 0 and len(cont_ms) == 0), \
                str(cont_ms) + str(cont_swda)

            if len(cont_swda) > len(cont_ms):
                # print 'CONT case1'
                # SWDA script has more words, distribute length
                shared_length = end_time - current_MS_start_times[i]
                number_of_letters = len(reduce(lambda x, y: x + y, cont_swda))
                av_length = shared_length / float(number_of_letters)
                current_time = current_MS_start_times[i]
                for ins_word in cont_swda:
                    final_SWDA_words.append(ins_word)
                    final_start_times.append(current_time)
                    final_time = current_time + (av_length * len(ins_word))
                    final_end_times.append(final_time)
                    current_time = final_time
            elif len(cont_ms) > len(cont_swda):
                # print 'CONT case2'
                # MS words has more words, generalize the time over
                # shorter SWDA stretch
                shared_length = end_time - current_MS_start_times[i]
                number_of_letters = len(reduce(lambda x, y: x + y, cont_ms))
                av_length = shared_length / float(number_of_letters)
                current_time = current_MS_start_times[i]
                for ins_word in cont_swda:
                    final_SWDA_words.append(ins_word)
                    final_start_times.append(current_time)
                    final_time = current_time + (av_length * len(ins_word))
                    final_end_times.append(final_time)
                    current_time = final_time
            else:
                pass
                # print 'CONT case 3'

        else:
            # just a normal linking alignment if straight SUB/match
            if "</laughter>" in current_MSwords[i][0] and \
                    not "<laughter" in current_MS_SWDAwords[i] and \
                    not clean_word(current_MS_SWDAwords[i]) == "":
                current_MS_SWDAwords[i] = "<laughter>" + current_MS_SWDAwords[i] +\
                    "</laughter>"

            final_SWDA_words.append(current_MS_SWDAwords[i])
            final_start_times.append(current_MS_start_times[i])
            final_end_times.append(current_MS_end_times[i])

    while len(final_start_times) < len(final_SWDA_words):
        final_SWDA_words.pop()  # clear this out
    return final_SWDA_words, final_start_times, final_end_times,\
        final_delete, final_insert


def clean_word(word):
    if "[laughter]" in word:
        word = "<laughter/>"  # laughter bout
        return word
    if "[laughter-" in word:
        # TODO LAUGHTER WORD
        word = word.replace("[laughter-", "")[:-1]
        word = "<laughter>" + word + "</laughter>"
        return word
    if "[" in word and "/" in word:
        # TODO our vernacular dictionary #for now using the actually heard
        # word?
        word = word[word.find("/") + 1:-1]  # [arear/area]
    if "_1" in word:
        word = word.replace("_1", "")
    if "]-" in word:
        word = word[:word.find("[")] + "-"
        # TODO PARTIAL WORD
    if "-[" in word:
        word = "-" + word[word.find("]") + 1:]
    if "[" in word or "]" in word or "_" in word:
        word = word.replace("[", "").replace("]", "").replace("_", "")
    return word


def map_MS_to_SWDA(MSfilename, SWDAindices, SWDAwords, laughter=False,
                   debug=False):
    """For given dialogue speaker (and MS file):
        Align in two stages -
        1) Use the edits marked up by the MS transcribers (6th column)
        to map from the MS transcription (7th column) \
        the old SWDA transcriptions (5th column)
        2) align the old SWDA transcriptions (5th column) with
        the STIR disfluency mark-up
        So then with these two transitive mappings in place, ouput
        the best link the MS words + timings
        to the SWDA transcription indices, such that all MS words have a link,
        as do all SWDA transcription indices.

    From these alignments, get the words, tags and timings in the
    following ways:

    CURRENT METHOD (favours the original SWDA transcription words/number
    of tokens but gets MS timings):
        Map the SWDA words to the MS time-aligned words to get the timings
        for each SWDA word.
        When MS words must be DELeted to get back to the original
        transcriptions:
        - if contiguous (no gap) to the previous word make the original
        word adopt the end time of the deleted word
        - else just ignore these words in final mark-up
        When MS words must be INS to get back to the original
        SWDA transcription, i.e. no word timings:
        - Simply divide the time of the previous matching word in
        terms of the number of letters in each word, i.e.
        'uh yeah' <ins> yeah 0 0.6 would get (uh 0 0.2, and yeah 0.2 0.6)
        not ideal, but good enough for now

    FUTURE METHOD (favours the MS new transcriptions):
        (this may need to be verified by hand for the new disfluency mark-up)
        When words are DELeted to get back to the SWDA ones, these will by
        default have <f/> tags, unless
        - in {uh/um/uh-huh/um-hum/oh} (edit terms) or if not utterance
        final and they are partial, they will be
        reparandum words with a new delete onset
        -else, if they are between reparandum words or at the end of a
        reparandum before an edit), they adopt reparandum tags,
        -else if they are at the end of a reparandum they can be repair starts!
        (difficult to tell)
        -else if they are between repair words, they adopt repair tags

        When words are INSerted to get back to the SWDA ones
        (i.e. the original transcribers hallucinate a word)
        -if they are the sole repair word in a repair, then that
        whole repair gets deleted
    """
    MSfile = open(MSfilename)
    # print MSfilename
    #print (len(SWDAindices),len(SWDAwords))

    start_times = []  # all start times of MS words
    end_times = []  # all end times of MS words
    words = []  # a list of the MS words being mapped,
    laughter_labels = []  # all the laughter events in triples of status,b,e
    # will output these and maywell swap them in the final data

    count = 0
    # this goes up for each speaker's turn, increments as we go along
    new_SWDA_index = ""
    currentUttIndex = ""
    current_SWDAwords = []
    current_SWDAindices = []
    # will be tuples of (MSword, MS2_SWDA_infil_alignment)
    current_MSwords = []
    current_MS_SWDAwords = []
    current_start_times = []
    current_end_times = []
    final_insert_words = None
    SWDA_left_over = None
    previous_alignment = ""
    continuation_number = 0  # nb for ensuring at least 2 consec. contin tags.

    # loop through, doing both mapping stages at the end of an utterance
    # according to utterance index at column 1
    for line in MSfile:
        data = line.split("\t")
        assert(len(data) == 7), data
        if debug:
            print data
            print currentUttIndex

        # 1. check for bad annotations
        if "<DEL>" in data[4] and continuation_number == 1:
            print "ANNOTATION ERROR, changing from DEL to CONT"
            print "\t" + str(data)
            data[4] = "<CONT>"
            print "\t" + "changed to"
            print "\t" + str(data)
        if "<DEL>" in data[4] and not "---" in data[5]:
            # account for incorrect labels: e.g. <DEL>    Uh-huh    um-hum
            print "ANNOTATION ERROR, changing from DEL to SUB"
            print "\t" + str(data)
            data[4] = "<SUB>"
            print "\t" + "changed to"
            print "\t" + str(data)
        if "<INS>" in data[4] and not "---" in data[6]:
            print "ANNOTATION ERROR, changing from INS to SUB"
            print "\t" + str(data)
            data[4] = "<SUB>"
            print "\t" + "changed to"
            print "\t" + str(data)
        if "<CONT>" in data[4]:
            continuation_number += 1
        else:
            continuation_number = 0

        # 2. check for skipping in either one and whether this is a laughter
        # bout or not
        laughter_status = False
        MSskip = False
        if "[silence]" in data[6] or "noise]" in data[6]:
            MSskip = True
        elif "[laughter]" in data[6] or "<e_aside>" in data[6] \
                or "<b_aside>" in data[6]:
            # laughter bout a special case
            if "[laughter]" in data[6] and laughter:
                if debug:
                    print "MS laughter turn"
                laughter_status = True
            else:
                MSskip = True
        elif "+++" in data[6] or "---" in data[6]:
            # delete or continuation in SWDA version
            MSskip = True

        SWDAskip = False
        if "[silence]" in data[5] or "noise]" in data[5]:
            if debug:
                print "swdaskip silence or noise"
            SWDAskip = True
        elif "[laughter]" in data[5] or "<e_aside>" in data[5] or\
                "<b_aside>" in data[5]:
            # TODO Laughter- is it always marked up in old swda?
            if "[laughter]" in data[5]:
                if debug:
                    print "SWDA laughter turn"
                pass
            else:
                if debug:
                    print "swdaskip silence or noise"
                SWDAskip = True
        elif "+++" in data[5] or "---" in data[5]:
            # delete or continuation in MS version, already detected
            if debug:
                print "swdaskip silence or noise"
            SWDAskip = True

        # 3. get the data
        MSSWDAalignment = data[4]
        current_start_time = float(data[2].strip())
        word = data[5].replace("\n", "")  # take the SWDA word to be mapped
        # the MSword to be mapped to the SWDA
        MSword = data[6].replace("\n", "")
        current_end_time = float(data[3].strip())  # always resets
        if laughter_status:
            if current_SWDAindices:
                laughter_idx = current_SWDAindices[0]
            elif new_SWDA_index != "":
                laughter_idx = new_SWDA_index
            else:
                laughter_idx = SWDAindices[0]
                # print data
                #raise Exception
            laughter_labels.append((current_start_time,
                                    current_end_time,
                                    laughter_idx))
            continue

        if "[laughter-" in word:
            # TODO LAUGHTER WORD
            word = word.replace("[laughter-", "")[:-1]
            word = "<laughter>" + word + "</laughter>"
            if debug:
                print "word", word
        if "[laughter-" in MSword:
            # TODO LAUGHTER WORD
            MSword = MSword.replace("[laughter-", "")[:-1]
            MSword = "<laughter>" + MSword + "</laughter>"
            if debug:
                print "MSword", MSword
                # raw_input()
        # if utterance changes, add the last one
        if data[1].split(".")[1] != currentUttIndex and \
                not (MSSWDAalignment == "<CONT>" and
                     previous_alignment == "<CONT>"):
            # NB annoying thing that continuations are done across boundaries-
            # here just carry on,
            # treating it as part of the last utt.
            if debug:
                if debug:
                    print "adding last one"
            if currentUttIndex != "":
                # not the first time as the previous one doesn't exist
                # 1st alignment- returns lists the same length as
                # current_MS_SWDAwords with the best
                # approximation to the word start and end times for
                # those based on the MS timings/transcripts
                if debug:
                    print "*" * 8, "current before first adjust"
                    print len(current_MSwords), len(current_MS_SWDAwords)
                    print len(current_start_times), len(current_end_times)
                    print current_MSwords, current_MS_SWDAwords
                    print "*" * 4
                current_MS_SWDAwords,\
                    current_start_times,\
                    current_end_times,\
                    _,\
                    final_insert_words = timings_for_MS_SWDAwords_from_MSwords(
                        current_MSwords,
                        current_MS_SWDAwords,
                        current_start_times,
                        current_end_times,
                        data=data)
                if debug:
                    print "*" * 8, "current after first adjust"
                    print len(current_MSwords), len(current_MS_SWDAwords)
                    print len(current_start_times), len(current_end_times)
                    print current_MSwords, current_MS_SWDAwords
                    print "*" * 4
                # print current_MSwords, current_MS_SWDAwords
                # 2nd alignment- produces a mapping the same
                # length as current_MS_SWDAwords
                # which is in the format of the MS files
                # links to the words in SWDA words
                if current_MS_SWDAwords != []:  # must be words to add
                    temp_current_MS_SWDAwords, temp_current_SWDAwords,\
                        current_start_times,\
                        current_end_times,\
                        alignment = get_best_word_alignment(
                            current_MS_SWDAwords,
                            current_SWDAwords,
                            current_start_times,
                            current_end_times)

                    if debug:
                        print "*" * 8, "after alignment"
                        print temp_current_MS_SWDAwords
                        print temp_current_SWDAwords
                        print current_start_times
                        print current_end_times
                        print current_SWDAindices
                        print "*" * 8
                    # redo the start times again, this time the length of
                    # SWDA words, with the right approximations?
                    temp_current_SWDAwords, current_start_times, \
                        current_end_times, _, _ = \
                        timings_for_MS_SWDAwords_from_MSwords(
                            temp_current_MS_SWDAwords,
                            temp_current_SWDAwords,
                            current_start_times,
                            current_end_times,
                            data=data)
                    if debug:
                        print "*" * 8, "after re-adjust"
                        print temp_current_SWDAwords
                        print current_start_times
                        print current_end_times
                        print current_SWDAindices  # what happens to these?
                        print "*" * 8

                    assert current_SWDAwords == [x.replace("<laughter>", "").
                                                 replace("</laughter>", "")
                                                 for x in temp_current_SWDAwords],\
                        str(temp_current_SWDAwords) + "\n" + \
                        str(current_SWDAwords)
                    # add the laughter back in
                    current_SWDAwords = temp_current_SWDAwords
                    # mappingToSWDA.extend(list(current_mappings))
                    assert len(current_start_times) == len(current_end_times)\
                        == len(current_SWDAwords) == len(current_SWDAindices)
                    start_times.extend(current_start_times)
                    end_times.extend(current_end_times)
                    words.extend(current_SWDAwords)
                else:
                    if debug:
                        print "MSSWDA words empty"
                    SWDA_left_over = deepcopy(current_SWDAwords)
                    SWDA_left_over_indices = deepcopy(current_SWDAindices)

            # get new utt index
            if not data[1].split(".")[1] == currentUttIndex:

                currentUttIndex = data[1].split(".")[1]
                if debug:
                    print "utt Index now", currentUttIndex

            current_start_times = []
            current_end_times = []
            current_MSwords = []
            current_MS_SWDAwords = []
            current_SWDAwords = []
            current_SWDAindices = []
            if final_insert_words:  # extend any extras at the beginning
                # print "final insert words found", final_insert_words
                current_MS_SWDAwords.extend(deepcopy(final_insert_words))
                current_MSwords.extend(
                    len(final_insert_words) * [("", "<INS>")])
                dummy_start_time = 0
                dummy_end_time = current_start_time
                if len(end_times) > 0:  # use the last value if it exists
                    dummy_start_time = end_times[-1]
                    dummy_end_time = end_times[-1]
                current_start_times.extend(len(final_insert_words)
                                           * [dummy_start_time])
                current_end_times.extend(len(final_insert_words)
                                         * [dummy_end_time])
                if SWDA_left_over:
                    current_SWDAwords = deepcopy(SWDA_left_over)
                    current_SWDAindices = deepcopy(SWDA_left_over_indices)
                    SWDA_left_over = None
                    SWDA_left_over_indices = None

                final_insert_words = None
            if SWDAindices != []:
                # print SWDAindices
                # to allow for continuations over boundaries, keep going
                # until up to current utterance
                while int(SWDAindices[0][1:-1].split(":")[0]) <= \
                        int(currentUttIndex):
                    # may have already been added in the 'skip'
                    new_SWDA_word = SWDAwords.pop(0)
                    new_SWDA_index = SWDAindices.pop(0)
                    # if debug:
                    #    print "adding WORD", new_SWDA_word, new_SWDA_index
                    current_SWDAwords.append(new_SWDA_word)  # currentSWDA[1]
                    current_SWDAindices.append(new_SWDA_index)
                    # if debug:
                    #    print "CURRENT SWDA", currentUttIndex,
                    #    print current_SWDAwords
                    if SWDAindices == []:
                        break
            if current_SWDAindices == []:
                if debug:
                    print "no swda words for", currentUttIndex, MSfilename
                pass

        if MSskip:
            MSword = ""
            if SWDAskip:
                if debug:
                    print "swda skip continue"
                continue  # don't deal with this as no word for either one
        else:
            MSword = clean_word(MSword)
        word = "" if SWDAskip else clean_word(word)

        # append the words and indices
        previous_alignment = MSSWDAalignment
        current_start_times.append(current_start_time)
        current_end_times.append(current_end_time)
        current_MS_SWDAwords.append(word)
        current_MSwords.append((MSword, MSSWDAalignment))
        count += 1

    # flush
    if current_MS_SWDAwords != []:
        temp_current_MS_SWDAwords, \
            temp_current_SWDAwords, \
            current_start_times, \
            current_end_times, \
            alignment = get_best_word_alignment(current_MS_SWDAwords,
                                                current_SWDAwords,
                                                current_start_times,
                                                current_end_times)
        # print "*"* 8, "after alignment"
        # print temp_current_MS_SWDAwords
        # print temp_current_SWDAwords
        # print current_start_times
        # print current_end_times
        # print current_SWDAindices
        # print "*"* 8
        # redo the start times again, this time the length of SWDA words,
        # with the right approximations?
        temp_current_SWDAwords,\
            current_start_times, \
            current_end_times,\
            _,\
            final_insert_words = timings_for_MS_SWDAwords_from_MSwords(
                temp_current_MS_SWDAwords,
                temp_current_SWDAwords,
                current_start_times,
                current_end_times,
                data=data)
        # print "*"* 8, "after re-adjust"
        # print temp_current_SWDAwords
        # print current_start_times
        # print current_end_times
        # print current_SWDAindices #what happens to these?
        # print "*"* 8
        assert current_SWDAwords == [x.replace("<laughter>", "").
                                     replace("</laughter>", "")
                                     for x in temp_current_SWDAwords],\
            str(temp_current_SWDAwords) + "\n" + \
            str(current_SWDAwords)
        # add the laughter back in
        current_SWDAwords = temp_current_SWDAwords
        assert len(current_start_times) == len(current_end_times) ==\
            len(current_SWDAwords) == len(current_SWDAindices)
        start_times.extend(current_start_times)
        end_times.extend(current_end_times)
        words.extend(current_SWDAwords)
    else:
        print "MSSWDA words empty"
        # one has to add on the remaining time
        SWDA_left_over = deepcopy(current_SWDAwords)
        SWDA_left_over_indices = deepcopy(current_SWDAindices)

        # divide length with the last word- not great
        overall_length = end_times[-1] - start_times[-1]
        av_length = overall_length / float(len(SWDA_left_over) + 1.0)
        # adjust the last word
        end_times[-1] = start_times[-1] + av_length
        for _ in SWDA_left_over:
            current = end_times[-1]
            start_times.append(current)
            end_times.append(current + av_length)
        words.extend(SWDA_left_over)
    MSfile.close()
    assert SWDAwords == [], str(SWDAwords)  # make sure they've all gone
    assert len(start_times) == len(end_times) == len(words), "Not even timings! \
                {} {} {}".format(len(start_times), len(end_times), len(words))
    return start_times, end_times, words, laughter_labels


def getSWDAspeaker(MSfilename):
    f = open(MSfilename)
    for line in f:
        data = line.split("\t")
        speaker = data[1].split(".")[0].replace("@", "")
        break
    f.close()
    return speaker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adds word timings\
    to disfluency detection file.')
    parser.add_argument('-i', action='store', dest='corpusLocation',
                        default='../data/disfluency_detection/\
                        switchboard/swbd_disf_train_1_2_partial_data.csv',
                        help='location of the SWDA corpus file')
    parser.add_argument('-a', action='store', dest='wordAlignmentFolder',
                        default='./data/raw_data/swbd_alignments',
                        help='location of the word alignment files')
    parser.add_argument('-l', action='store_true', dest='annotateLaughter',
                        default=False)
    parser.add_argument('-d', action='store', dest='divisionFile',
                        default='../data/disfluency_detection/\
                        swda_divisions_disfluency_detection/\
                        SWDisfTrain_ranges.text',
                        help='location of the file listing the \
                        files used in the corpus')

    args = parser.parse_args()
    range_file = args.divisionFile
    disf_file = args.corpusLocation
    # path to the MS aligned transcripts
    alignments_dir = args.wordAlignmentFolder
    write_mapping = True
    # "../../../swbd/MSaligned/alignments"
    # nb don't really need the below, just for writing map
    if write_mapping:
        timings_corpus_file = open(args.corpusLocation.replace("data.csv",
                                                               "data_timings.csv"), "w")
        # path to the mappings folder we want to write the mapping files to
        # this can be temp and then removed.
        # mapping_dir = os.path.dirname(os.path.realpath(__file__)) +\
        #                "/../data/raw_data/swbd_swda2MS_mapping_temp"
        # make that dir if it doesn't exist
        # if not os.path.isdir(mapping_dir):
        #    os.mkdir(mapping_dir)

    ranges = sorted([line.strip("\n") for line in open(range_file)])

    print len(ranges), "files to process"

    dialogue_speakers = []
    # for disf_file in disfluency_files:
    IDs, mappings, utts, pos_tags, labels = \
        load_data_from_disfluency_corpus_file(disf_file)
    # print labels
    dialogue_speakers.extend(sort_into_dialogue_speakers(IDs, mappings, utts,
                                                         pos_tags, labels,
                                                         convert_to_dnn_tags=False))
    print len(dialogue_speakers), "dialogue speakers"
    # The main loop- has every word in both formats, needs to map from MS file
    # timings to the SWDA ones
    # Given the original SWDA transcripts are IN the MSaligned files, it's safer
    # to map to them,
    # then use the delete/insert/sub operations in the files to get the pointer.
    # So the mapping is MSnew -> MSSWDAold -> SWDAold,
    # where the last mapping is done through min. edit distance string
    # alignment
    tests = ["2549B"]
    print "Creating word timing aligned corpus..."
    for dialogue_triple in sorted(dialogue_speakers, key=lambda x: x[0]):
        dialogue_speakerID, SWDAindices, SWDAwords, SWDApos, SWDAlabels = \
            dialogue_triple
        origSWDAindices = deepcopy(SWDAindices)
        origSWDAwords = deepcopy(SWDAwords)

        # print dialogue_speakerID
        # print SWDAindices
        # print SWDAwords
        # print SWDApos
        # print SWDAlabels
        # if not dialogue_speakerID in tests:
        #    continue
        # if int(dialogue_speakerID[:4]) < int(tests[0][:4]):
        #    continue
        # print SWDAindices,
        # print SWDAwords
        MSID = dialogue_speakerID[-1]
        # print SWDAindices[0:50], len(SWDAindices)
        # print SWDAwords[0:50], len(SWDAwords)
        MSfilename = alignments_dir + "/" + dialogue_speakerID[0:1] + "/" +\
            "sw{}-ms98-a-penn.text".format(dialogue_speakerID)
        # can switch between A and B, use the original PTB role names
        # (not the MS ones)
        speaker = getSWDAspeaker(MSfilename)
        # print speaker
        # NB do we need to sample the first n words to check which speaker
        # is which???
        if not speaker in MSID and not "2434" in MSfilename:
            # NB 2434 is wrong in both versions
            switched_speakerID = dialogue_speakerID[:-1] + speaker
            MSfilename = alignments_dir + "/" + dialogue_speakerID[0:1] + "/" +\
                "sw{}-ms98-a-penn.text".format(switched_speakerID)

        start_times, stop_times, SWDAwords, laughter_bouts = map_MS_to_SWDA(
            MSfilename,
            SWDAindices,
            SWDAwords,
            laughter=args.annotateLaughter)

        if not len(origSWDAwords) == len(start_times):
            c = 0
            print "ERROR uneven lengths!", len(origSWDAwords), len(SWDAwords)
            for x, y in zip(origSWDAwords, SWDAwords):
                if x != y:
                    print x, y
                c += 1
            print dialogue_speakerID
            raise Exception
        # break
        if write_mapping:
            timings_corpus_file.write("Speaker: " + dialogue_speakerID + "\n")
            # mapping_file = open(mapping_dir + "/" + dialogue_speakerID +\
            #                     ".csv","w")
            # print laughter_bouts
            last_time_stamp = 0
            for start, stop, indices, word, swda_word, pos_tag, label in \
                zip(start_times,
                    stop_times,
                    origSWDAindices,
                    origSWDAwords,
                    SWDAwords,
                    SWDApos,
                    SWDAlabels):
                # print start, stop, indices, word
                if "<laughter>" in swda_word:
                    #word = "<laughter>" + word + "</laughter>"
                    label += "<speechLaugh/>"
                # mapping_file.write("\t".join([str(start),str(stop),\
                #                              indices,word])+\
                #                   "\n")
                # if laughter_bouts and laughter_bouts[0][0]<start:
                #    print laughter_bouts[0], start, stop, last_time_stamp
                #    raw_input()
                if laughter_bouts and laughter_bouts[0][0] < start and\
                        laughter_bouts[0][0] >= last_time_stamp:
                    bout = laughter_bouts.pop(0)

                    assert(bout[1] <= stop), "laughter {0} {1} {2}".format(bout,
                                                                           dialogue_speakerID,
                                                                           stop)
                    timings_corpus_file.write("\t".join([bout[2],
                                                         str(bout[0]),
                                                         str(bout[1]),
                                                         "<laughter/>",
                                                         "LAUGHTER",
                                                         "<laughter/>"]) + "\n")

                timings_corpus_file.write("\t".join([indices,
                                                     str(start),
                                                     str(stop),
                                                     word,
                                                     pos_tag,
                                                     label]) + "\n")
                last_time_stamp = stop
            timings_corpus_file.write("\n")
            # mapping_file.close()

    if write_mapping:
        timings_corpus_file.close()
    print "Timing aligned corpus complete."
