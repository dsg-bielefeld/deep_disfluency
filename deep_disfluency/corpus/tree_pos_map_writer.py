# -*- coding: utf-8 -*-
from collections import defaultdict
import re
from nltk import tree

from swda import CorpusReader
from tree_pos_map import TreeMapCorpus
from tree_pos_map import POSMapCorpus

possibleMistranscription = [("its", "it's"),
                            ("Its", "It's"),
                            ("it's", "its"),
                            ("It's", "Its"),
                            ("whose", "who's"),
                            ("Whose", "Who's"),
                            ("who's", "whose"),
                            ("Who's", "Whose"),
                            ("you're", "your"),
                            ("You're", "Your"),
                            ("your", "you're"),
                            ("Your", "You're"),
                            ("their", "they're"),
                            ("Their", "They're"),
                            ("they're", "their"),
                            ("They're", "Their"),
                            ("programme", "program"),
                            ("program", "programme"),
                            ("centre", "center"),
                            ("center", "centre"),
                            ("travelling", "traveling"),
                            ("traveling", "travelling"),
                            ("colouring", "coloring"),
                            ("coloring", "colouring")]


class TreeMapWriter:
    """Object which writes mappings from the words in utterances
    to the nodes of the corresponding trees in a treebank
    """

    def __init__(self, corpus_path="../swda",
                 metadata_path="swda-metadata.csv",
                 target_folder_path="Maps",
                 ranges=None,
                 errorLog=None):
        print "started TreeMapWriting"
        self.write_to_file(corpus_path,
                           metadata_path,
                           target_folder_path,
                           ranges,
                           errorLog)

    def write_to_file(self, corpus_path,
                      metadata_path,
                      target_folder_path,
                      ranges,
                      errorLog):
        """Writes files to a target folder with the mappings
        from words in utterances to tree nodes in trees.
        """

        if errorLog:
            errorLog = open(errorLog, 'w')
        corpus = CorpusReader(corpus_path, metadata_path)
        # Iterate through all transcripts
        incorrectTrees = 0
        folder = None
        corpus_file = None

        for trans in corpus.iter_transcripts():

            # print "iterating",trans.conversation_no
            if not trans.has_pos():
                continue
            # print "has pos"
            if ranges and not trans.conversation_no in ranges:
                continue
            # print "in range"
            # just look at transcripts WITH trees as compliment to the
            # below models
            if not trans.has_trees():
                continue
            end = trans.swda_filename.rfind("/")
            start = trans.swda_filename.rfind("/", 0, end)
            c_folder = trans.swda_filename[start + 1:end]
            if c_folder != folder:
                # for now splitting the maps by folder
                folder = c_folder
                if corpus_file:
                    corpus_file.close()
                corpus_file = open(target_folder_path +
                                   "/Tree_map_{0}.csv.text".format(folder), 'w')
                wordTreeMapList = TreeMapCorpus(False, errorLog)
                print "new map for folder", folder

            translist = trans.utterances
            translength = len(translist)
            count = 0

            # iterating through transcript utterance by utterance
            # create list of tuples i.e. map from word to the index(ices)
            #(possibly multiple or null) of the relevant leaf/ves
            # of a given tree i.e. utt.tree[0].leaves[0] would be a pair (0,0))
            while count < translength:
                utt = trans.utterances[count]
                words = utt.text_words()
                wordTreeMap = []  # [((word), (List of LeafIndices))]
                forwardtrack = 0
                backtrack = 0
                continued = False
                # print "\n COUNT" + str(count)
                # print utt.damsl_act_tag()
                if len(utt.trees) == 0 or utt.damsl_act_tag() == "x":
                    wordTreeMap.append((utt, []))  # just dummy value
                    # errormessage = "WARNING: NO TREE for file/utt: " +\
                    # str(utt.swda_filename) + " " + utt.caller + "." +  \
                    # str(utt.utterance_index) + "." + \
                    #str(utt.subutterance_index) + " " + utt.text
                    # print(errormessage)
                    count += 1
                    continue
                    # raw_input()

                # indices for which tree and leaf we're at:
                i = 0  # tree
                j = 0  # leaf
                # initialise pairs of trees and ptb pairs
                trees = []
                for l in range(0, len(utt.trees)):
                    trees.append((utt.ptb_treenumbers[l], count, l, utt.trees[l]))
                # print "TREES = "
                # for tree in trees:
                #    print tree
                origtrees = list(trees)
                origcount = count
                # overcoming the problem of previous utterances contributing
                # to the tree at this utterance, we need to add the words from
                # the previous utt add in all the words from previous utterance
                # with a dialogue act tag/or the same tree?
                # check that the last tree in the previous utterance
                # is the same as the previous one
                previousUttSame = trans.previous_utt_same_speaker(utt)
                # print previousUttSame
                lastTreeMap = None
                if previousUttSame:
                    # print "search for previous full act utt
                    # for " + str(utt.swda_filename) + str(utt.transcript_index)
                    lastTreeMap = wordTreeMapList.get_treemap(
                        trans,
                        previousUttSame)
                    if ((not lastTreeMap) or (len(lastTreeMap) == 0) or
                            (len(lastTreeMap) == 1 and lastTreeMap[0][1] == [])):
                        # print "no last tree map, backwards searching"
                        while previousUttSame and \
                            ((not lastTreeMap) or (len(lastTreeMap) == 0) or
                             (len(lastTreeMap) == 1 and lastTreeMap[0][1] == [])):
                            previousUttSame = trans.previous_utt_same_speaker(
                                previousUttSame)  # go back one more
                            lastTreeMap = wordTreeMapList.get_treemap(trans,
                                                                      previousUttSame)
                            if previousUttSame:
                                pass
                                # print previousUttSame.transcript_index

                    if not lastTreeMap:
                        pass
                        # print "no last treemap found for:"
                        # print utt.swda_filename
                        # print utt.transcript_index

                if lastTreeMap and \
                        (utt.damsl_act_tag() == "+" or
                         (len(lastTreeMap.treebank_numbers) > 0
                          and lastTreeMap.treebank_numbers[-1] ==
                          utt.ptb_treenumbers[0])):
                    continued = True
                    # might have to backtrack
                    # now checking for wrong trees
                    lastPTB = lastTreeMap.treebank_numbers
                    lastIndexes = lastTreeMap.transcript_numbers
                    lastTreesTemp = lastTreeMap.get_trees(trans)
                    lastTrees = []
                    for i in range(0, len(lastPTB)):
                        lastTrees.append([lastPTB[i], lastIndexes[i][0],
                                          lastIndexes[i][1], lastTreesTemp[i]])
                    if not (lastPTB[-1] == utt.ptb_treenumbers[0]):
                        # print "not same, need to correct!"
                        # print words
                        # print trees
                        # print "last one"
                        # print previousUttSame.text_words()
                        # print lastTrees
                        if utt.ptb_treenumbers[0] - lastPTB[-1] > 1:
                            # backtrack and redo the antecedent
                            count = count - (count - lastIndexes[-1][0])
                            utt = previousUttSame
                            words = utt.text_words()
                            mytrees = []
                            for i in range(0, len(lastTrees) - 1):
                                mytrees.append(lastTrees[i])
                            trees = mytrees + [origtrees[0]]
                            # print "\n(1)backtrack to with new trees:"
                            backtrack = 1
                            # print utt.transcript_index
                            # print words
                            # print trees
                            # raw_input()
                        # alternately, this utt's tree may be further back
                        # than its antecdent's, rare mistake
                        elif utt.ptb_treenumbers[0] < lastTrees[-1][0]:
                            # continue with this utterance and trees
                            #(if there are any), but replace its first
                            # tree with its antecdents last one
                            forwardtrack = 1
                            trees = [lastTrees[-1]] + origtrees[1:]
                            # print "\n(2)replacing first one to lasttreemap's:"
                            # print words
                            # print trees
                            # raw_input()

                    if backtrack != 1:  # we should have no match
                        found_treemap = False
                        # resetting
                        # for t in wordTreeMapList.keys():
                        #        print t
                        #        print wordTreeMapList[t]
                        for t in range(len(lastTreeMap) - 1, -1, -1):
                            # print lastTreeMap[t][1]
                            # if there is a leafIndices for the
                            # word being looked at, gets last mapped one
                            if len(lastTreeMap[t][1]) > 0:
                                # print "last treemapping of last
                                # caller utterance =
                                #" + str(lastTreeMap[t][1][-1])
                                j = lastTreeMap[t][1][-1][1] + 1
                                found_treemap = True
                                # print "found last mapping, j -1 = " + str(j-1)
                                # raw_input()
                                break
                        if not found_treemap:
                            pass
                            # print "NO matched last TREEMAP found for \
                            # previous Utt Same Speaker of " + \
                            # str(trans.swda_filename) + " " + \
                            # str(utt.transcript_index)
                            # print lastTreeMap
                            # for tmap in wordTreeMapList.keys():
                            #    print tmap
                            #    print wordTreeMapList[tmap]
                            # raw_input()

                possibleComment = False  # can have comments, flag
                mistranscribe = False
                LeafIndices = []  # possibly empty list of leaf indices
                word = words[0]
                # loop until no more words left to be matched in utterance
                while len(words) > 0:
                    # print "top WORD:" + word
                    if not mistranscribe:
                        wordtest = re.sub(r"[\.\,\?\"\!]", "", word)
                        wordtest = wordtest.replace("(", "").replace(")", "")
                    match = False
                    LeafIndices = []  # possibly empty list of leaf indices
                    if (possibleComment
                        or word[0:1] in ["{", "}", "-"]
                        or word in ["/", ".", ",", "]"]
                        or wordtest == ""
                        or any([x in word for x in ["<", ">", "*", "[", "+", "]]",
                                                    "...", "#", "="]])):
                        # no tree equivalent for {D } type annotations
                        if (word[0:1] == "-" or
                                any([x in word for x in
                                     ["*", "<<", "<+", "[[", "<"]])) \
                                and not possibleComment:
                            possibleComment = True
                        if possibleComment:
                            #print("match COMMENT!:" + word)
                            # raw_input()
                            LeafIndices = []
                            match = True
                            #wordTreeMap.append((word, LeafIndices))
                            if any([x in word for x in [">>", "]]", ">"]]) or \
                                    word[0] == "-":  # turn off comment
                                possibleComment = False
                                #del words[0]
                        # LeadIndices will be null here
                        wordTreeMap.append((word, LeafIndices))
                        LeafIndices = []
                        match = True
                        # print "match annotation!:" + word
                        del words[0]  # word is consumed, should always be one
                        if len(words) > 0:
                            word = words[0]
                            wordtest = re.sub(r"[\.\,\?\/\)\(\"\!]", "", word)
                            wordtest = wordtest.replace("(", "")
                            wordtest = wordtest.replace(")", "")
                        else:
                            break
                        continue
                        # carry on to next word without updating indices?
                    else:
                        while i < len(trees):
                            # print "i number of trees :" + str(len(utt.trees))
                            # print "i tree number :" + str(i)
                            # print "i loop word :" + word
                            tree = trees[i][3]
                            # print "looking at ptb number " + str(trees[i][0])
                            # print "looking at index number " \
                            #+ str(trees[i][1])+","+str(trees[i][2])
                            while j < len(tree.leaves()):
                                leaf = tree.leaves()[j]
                                # print "j number of leaves : " \
                                #+ str(len(tree.leaves()))
                                # print "j loop word : " + word
                                # print "j loop wordtest : " + wordtest
                                # print "j leaf : " + str(j) + " " + leaf
                                breaker = False
                                # exact match
                                if wordtest == leaf or word == leaf:
                                    LeafIndices.append((i, j))
                                    wordTreeMap.append((word, LeafIndices))
                                    # print("match!:" + word + " " + \
                                    # str(utt.swda_filename) + " " + \
                                    # utt.caller + "." +  \
                                    # str(utt.utterance_index) + \
                                    #"." + str(utt.subutterance_index))
                                    del words[0]  # word is consumed
                                    if len(words) > 0:
                                        word = words[0]  # next word
                                        wordtest = re.sub(
                                            r"[\.\,\?\/\)\(\"\!]", "", word)
                                        wordtest = wordtest.replace("(", "")
                                        wordtest = wordtest.replace(")", "")
                                    LeafIndices = []
                                    j += 1  # increment loop to next leaf
                                    match = True
                                    breaker = True
                                    # raw_input()
                                    break
                                elif leaf in wordtest or \
                                        leaf in word and not leaf == ",":
                                    testleaf = leaf
                                    LeafIndices.append((i, j))
                                    j += 1
                                    for k in range(j, j + 3):  # 3 beyond
                                        if (k >= len(tree.leaves())):
                                            j = 0
                                            i += 1
                                            #breaker = True
                                            breaker = True
                                            break  # got to next tree
                                        if (testleaf + tree.leaves()[k]) \
                                                in wordtest or (testleaf +
                                                                tree.leaves()[k])\
                                                in word:
                                            testleaf += tree.leaves()[k]
                                            LeafIndices.append((i, k))
                                            j += 1
                                            # concatenation
                                            if testleaf == wordtest or \
                                                    testleaf == word:  # word matched
                                                wordTreeMap.append((word,
                                                                    LeafIndices))
                                                del words[0]  # remove word
                                                # print "match!:" + word +\
                                                #str(utt.swda_filename) + " "\
                                                # + utt.caller + "." +  \
                                                # str(utt.utterance_index) +\
                                                # "." + \
                                                # str(utt.subutterance_index))
                                                if len(words) > 0:
                                                    word = words[0]
                                                    wordtest = re.sub(
                                                        r"[\.\,\?\/\)\(\"\!]",
                                                        "", word)
                                                    wordtest = wordtest.\
                                                        replace("(", "")
                                                    wordtest = wordtest.\
                                                        replace(")", "")
                                                # reinitialise leaves
                                                LeafIndices = []
                                                j = k + 1
                                                match = True
                                                breaker = True
                                                # raw_input()
                                                break
                                else:
                                    # otherwise go on
                                    j += 1
                                if breaker:
                                    break
                                if match:
                                    break
                            if j >= len(tree.leaves()):
                                j = 0
                                i += 1
                            if match:
                                break

                    # could not match word! try mistranscriptions first:
                    if not match:
                        if not mistranscribe:  # one final stab at matching!
                            mistranscribe = True
                            for pair in possibleMistranscription:
                                if pair[0] == wordtest:
                                    wordtest = pair[1]
                                    if len(wordTreeMap) > 0:
                                        if len(wordTreeMap[-1][1]) > 0:
                                            i = wordTreeMap[-1][1][-1][0]
                                            j = wordTreeMap[-1][1][-1][1]
                                        else:
                                            # go back to beginning of
                                            # tree search
                                            i = 0
                                            j = 0
                                    else:
                                        i = 0  # go back to beginning
                                        j = 0
                                    break  # matched
                        elif continued:
                            # possible lack of matching up of words in
                            # previous utterance same caller and same
                            # tree// not always within same tree!!
                            errormessage = "Possible bad start for \
                            CONTINUED UTT ''" + words[0] + "'' in file/utt: "\
                                + str(utt.swda_filename) + "\n " + utt.caller + \
                                "." + str(utt.utterance_index) + "." + \
                                str(utt.subutterance_index) + \
                                "POSSIBLE COMMENT = " + str(possibleComment)
                            # print errormessage
                            if not errorLog == None:
                                errorLog.write(errormessage + "\n")
                            # raw_input()
                            if backtrack == 1:
                                backtrack += 1
                            elif backtrack == 2:
                                # i.e. we've done two loops and
                                # still haven't found it, try the other way
                                count = origcount
                                utt = trans.utterances[count]
                                words = utt.text_words()
                                word = words[0]
                                trees = [lastTrees[-1]] + origtrees[1:]
                                # print "\nSECOND PASS(2)replacing \
                                # first one to lasttreemap's:"
                                # print words
                                # print trees
                                backtrack += 1
                                # mistranscribe = False #TODO perhaps needed
                                wordTreeMap = []
                                # switch to forward track this is
                                # the only time we want to try
                                # from the previous mapped leaf in the
                                # other tree
                                foundTreemap = False
                                for t in range(len(lastTreeMap) - 1, -1, -1):
                                    # backwards iteration through words
                                    # print lastTreeMap[t][1]
                                    if len(lastTreeMap[t][1]) > 0:
                                        # print "last treemapping of last \
                                        # caller utterance = " + \
                                        # str(lastTreeMap[t][1][-1])
                                        j = lastTreeMap[t][1][-1][1] + 1
                                        foundTreemap = True
                                        # print "found last mapping, j = " \
                                        #+ str(j)
                                        # raw_input()
                                        # break when last tree
                                        # mapped word from this caller is found
                                        break
                                    if not foundTreemap:
                                        # print "NO matched last TREEMAP found\
                                        # for previous Utt Same Speaker of " + \
                                        # str(utt.swda_filename) + " " + \
                                        # utt.caller + "." +  \
                                        # str(utt.utterance_index) + "." +\
                                        #  str(utt.subutterance_index)
                                        j = 0
                                        # for tmap in wordTreeMapList.keys():
                                        #    print tmap
                                        #    print wordTreeMapList[tmap]
                                        # raw_input()
                                i = 0  # go back to first tree
                                continue
                            elif forwardtrack == 1:
                                forwardtrack += 1
                            elif forwardtrack == 2:
                                count = count - (count - lastIndexes[-1][0])
                                utt = previousUttSame
                                words = utt.text_words()
                                word = words[0]
                                mytrees = []
                                for i in range(0, len(lastTrees) - 1):
                                    mytrees.append(lastTrees[i])
                                trees = mytrees + [origtrees[0]]
                                # print "\nSECOND PASS(1)backtrack to \
                                # with new trees:"
                                # print utt.transcript_index
                                # print words
                                # print trees
                                forwardtrack += 1
                                # mistranscribe = False #TODO maybe needed
                                wordTreeMap = []
                                # raw_input()
                            elif forwardtrack == 3 or backtrack == 3:
                                # if this hasn't worked reset to old trees
                                # print "trying final reset"
                                count = origcount
                                utt = trans.utterances[count]
                                words = utt.text_words()
                                word = words[0]
                                trees = origtrees
                                forwardtrack = 0
                                backtrack = 0
                                # mistranscribe = False #TODO maybe needed
                                wordTreeMap = []
                                # raw_input()
                            else:
                                pass
                                # print "resetting search"
                                # raw_input()
                            # unless forward tracking now,
                            # just go back to beginning
                            i = 0  # go back to beginning of tree search
                            j = 0
                        else:
                            mistranscribe = False
                            LeafIndices = []
                            wordTreeMap.append((word, LeafIndices))
                            errormessage = "WARNING: 440 no/partial tree \
                            mapping for ''" + words[0] + "'' in file/utt: "\
                                + str(utt.swda_filename) + " \n" + utt.caller\
                                + "." + str(utt.utterance_index) + "." + \
                                str(utt.subutterance_index) + \
                                "POSSIBLE COMMENT = " + str(possibleComment)
                            # print utt.text_words()
                            del words[0]  # remove word
                            # for trip in wordTreeMap:
                            #    print "t",trip
                            if len(words) > 0:
                                word = words[0]
                                wordtest = re.sub(r"[\.\,\?\/\)\(\"\!]", "",
                                                  word)
                                wordtest = wordtest.replace("(", "")
                                wordtest = wordtest.replace(")", "")
                            # print errormessage
                            if errorLog:
                                errorLog.write("possible wrong tree mapping:"
                                               + errormessage + "\n")
                            raw_input()
                # end of while loop (words)
                mytreenumbers = []
                for treemap in trees:
                    # the whole list but the tree
                    mytreenumbers.append(treemap[:-1])
                if not len(utt.text_words()) == len(wordTreeMap):
                    print "ERROR. uneven lengths!"
                    print utt.text_words()
                    print wordTreeMap
                    print trans.swda_filename
                    print utt.transcript_index
                    raw_input()
                    count += 1
                    continue
                # add the treemap
                wordTreeMapList.append(trans.conversation_no,
                                       utt.transcript_index,
                                       tuple(mytreenumbers),
                                       tuple(wordTreeMap))
                count += 1
            # rewrite after each transcript
            filedict = defaultdict(str)
            for key in wordTreeMapList.keys():
                csv_string = '"' + str(list(wordTreeMapList[key])) + '"'
                mytreenumbers = wordTreeMapList[key].transcript_numbers
                myptbnumbers = wordTreeMapList[key].treebank_numbers
                tree_list_string = '"'
                for i in range(0, len(mytreenumbers)):
                    treemap = [myptbnumbers[i]] + mytreenumbers[i]
                    tree_list_string += str(treemap) + ";"
                tree_list_string = tree_list_string[:-1] + '"'
                filename = '"' + key[0:key.rfind(':')] + '"'
                transindex = key[key.rfind(':') + 1:]
                filedict[int(transindex)] = filename \
                    + "\t" + transindex + '\t' + csv_string + "\t" \
                    + tree_list_string + "\n"
            for key in sorted(filedict.keys()):
                corpus_file.write(filedict[key])

            wordTreeMapList = TreeMapCorpus(False, errorLog)  # reset each time
        print "\n" + str(incorrectTrees) + " incorrect trees"
        corpus_file.close()
        if not errorLog == None:
            errorLog.close()


class POSMapWriter:
    """Object which writes mappings from the words in utterances
    to the corresponding POS tags.
    """

    def __init__(self, corpus_path="../swda",
                 metadata_path="swda-metadata.csv",
                 target_folder_path="Maps",
                 ranges=None,
                 errorLog=None):
        print "started MapWriting"
        self.write_to_file(corpus_path,
                           metadata_path,
                           target_folder_path,
                           ranges,
                           errorLog)

    def write_to_file(self, corpus_path,
                      metadata_path,
                      target_folder_path,
                      ranges,
                      errorLog):
        """Writes files to a target folder with the mappings
        from words in utterances to corresponding POS tags.
        """
        if errorLog:
            errorLog = open(errorLog, 'w')
        corpus = CorpusReader(corpus_path, metadata_path)

        folder = None
        corpus_file = None
        for trans in corpus.iter_transcripts():

            # print "iterating",trans.conversation_no
            if not trans.has_pos():
                continue
            # print "has pos"
            if ranges and not trans.conversation_no in ranges:
                continue
            # print "in range"
            # just look at transcripts WITHOUT trees as compliment to the
            # above models
            if trans.has_trees():
                continue
            end = trans.swda_filename.rfind("/")
            start = trans.swda_filename.rfind("/", 0, end)
            c_folder = trans.swda_filename[start + 1:end]
            if c_folder != folder:
                # for now splitting the maps by folder
                folder = c_folder
                if corpus_file:
                    corpus_file.close()
                corpus_file = open(target_folder_path +
                                   "/POS_map_{0}.csv.text".format(folder), 'w')
                wordPOSMapList = POSMapCorpus(False, errorLog)
                print "new map for folder", folder

            translist = trans.utterances
            translength = len(translist)
            count = 0

            # iterating through transcript utterance by utterance
            while count < translength:
                utt = trans.utterances[count]
                words = utt.text_words()
                wordPOSMap = []
                if len(utt.pos) == 0:  # no POS
                    wordPOSMap.append((utt, []))  # just dummy value
                    wordPOSMapList.append(trans.conversation_no,
                                          utt.transcript_index,
                                          list(wordPOSMap))
                    errormessage = "WARNING: NO POS for file/utt: " +\
                        str(utt.swda_filename) + " " + utt.caller + "." + \
                        str(utt.utterance_index) + "." + \
                        str(utt.subutterance_index) + " " + utt.text
                    # print errormessage
                    # raw_input()
                else:
                    # indices for which POS we're at
                    j = 0
                    possibleComment = False  # can have comments, flag
                    mistranscribe = False
                    word = words[0]
                    # loop until no more words left to be matched in utterance
                    while len(words) > 0:
                        word = words[0]
                        # print "top WORD:" + word
                        if not mistranscribe:
                            wordtest = re.sub(r"[\.\,\?\/\)\(\"\!\\]", "",
                                              word)
                            wordtest = wordtest.replace("(", "").\
                                replace(")", "").replace("/", "")
                        match = False
                        POSIndices = []

                        if (possibleComment
                                or word[0:1] in ["{", "}", "-"]
                                or word in ["/", ".", ",", "]"]
                                or wordtest == ""
                                or any([x in word for x in
                                        ["<", ">", "*", "[", "+", "]]",
                                                   "...", "#", "="]])):
                            # no tree equivalent for {D } type annotations
                            if (word[0:1] == "-" or
                                    any([x in word for x in
                                         ["*", "<<", "<+", "[[", "<"]])) \
                                    and not possibleComment:
                                possibleComment = True
                            if possibleComment:
                                # print "match COMMENT!:" + word
                                # raw_input()
                                POSIndices = []
                                match = True
                                if (any([x in word for x in [">>", "]]", "))",
                                                             ">"]]) or
                                        word[0] == "-") \
                                        and not word == "->":
                                    # turn off comment
                                    possibleComment = False
                                if (">>" in word or "]]" in word or "))"
                                        in word or ">" in word and
                                        not word == "->"):  # turn off comment
                                    possibleComment = False
                                    #del words[0]
                            wordPOSMap.append((word, POSIndices))
                            POSIndices = []
                            match = True
                            # print "match annotation!:" + word
                            del words[0]  # word is consumed
                            if len(words) > 0:
                                word = words[0]
                                wordtest = re.sub(r"[\.\,\?\/\)\(\"\!\\]",
                                                  "", word)
                                wordtest = wordtest.replace("(", "")
                                wordtest = wordtest.replace(")", "")
                            else:
                                break
                            continue  # carry on to next word
                        else:
                            myPOS = utt.regularize_pos_lemmas()
                            while j < len(myPOS):
                                pos = myPOS[j][0]  # pair of (word,POS)
                                # print "j number of pos : " + str(len(myPOS))
                                # print "j loop word : " + word
                                # print "j loop wordtest : " + wordtest
                                # print "j pos : " + str(j) + " " + str(pos)
                                # raw_input()
                                breaker = False
                                if wordtest == pos or word == pos:  # exact match
                                    POSIndices.append(j)
                                    wordPOSMap.append((word, POSIndices))
                                    # print "match!:" + word + " in file/utt: "\
                                    # + str(utt.swda_filename) + \
                                    # str(utt.transcript_index))
                                    del words[0]  # word is consumed
                                    if len(words) > 0:
                                        word = words[0]  # next word
                                        wordtest = re.sub(
                                            r"[\.\,\?\/\)\(\"\!\\]",
                                            "", word)
                                        wordtest = wordtest.replace("(", "").\
                                            replace(")", "").replace("/", "")
                                    POSIndices = []
                                    j += 1  # increment lead number
                                    match = True
                                    breaker = True
                                    # raw_input()
                                    break
                                elif (pos in wordtest or pos in word) \
                                        and not pos in [",", "."]:
                                    # substring relation
                                    testpos = pos
                                    POSIndices.append(j)
                                    j += 1
                                    if wordtest[-1] == "-" and \
                                            pos == wordtest[0:-1]:
                                        wordPOSMap.append((word, POSIndices))
                                        del words[0]  # remove word
                                        # print "match!:" + word + " in \
                                        # file/utt: " + str(utt.swda_filename) \
                                        #+ str(utt.transcript_index)
                                        if len(words) > 0:
                                            word = words[0]
                                            wordtest = re.sub(
                                                r"[\.\,\?\/\)\(\"\!\\]",
                                                "", word)
                                            wordtest = wordtest.\
                                                replace("(", "").\
                                                replace(")", "").\
                                                replace("/", "")
                                            POSIndices = []
                                        match = True
                                        breaker = True
                                        break
                                    for k in range(j, j + 3):
                                        if (k >= len(myPOS)):
                                            breaker = True
                                            break
                                        if (testpos + myPOS[k][0]) in wordtest\
                                                or (testpos + myPOS[k][0]) in word:
                                            testpos += myPOS[k][0]
                                            POSIndices.append(k)
                                            j += 1
                                            # concatenation
                                            if testpos == wordtest or \
                                                    testpos == word:  # matched
                                                wordPOSMap.append((word,
                                                                   POSIndices))
                                                del words[0]  # remove word
                                                # print "match!:" +\
                                                # word + " in file/utt: " + \
                                                # str(utt.swda_filename) +\
                                                # str(utt.transcript_index))
                                                if len(words) > 0:
                                                    word = words[0]
                                                    wordtest = re.sub(
                                                        r"[\.\,\?\/\)\(\"\!\\]",
                                                        "", word)
                                                    wordtest = wordtest.\
                                                        replace("(", "")
                                                    wordtest = wordtest.\
                                                        replace(")", "")
                                                POSIndices = []
                                                j = k + 1
                                                match = True
                                                breaker = True
                                                break
                                else:
                                    j += 1  # otherwise go on
                                if breaker:
                                    break
                                if match:
                                    break

                        # could not match word! Could be mistransription
                        if not match:
                            # print "false checking other options"
                            # print j
                            # print word
                            # print wordtest
                            if not mistranscribe:
                                mistranscribe = True
                                for pair in possibleMistranscription:
                                    if pair[0] == wordtest:
                                        wordtest = pair[1]
                                        break  # matched
                                if wordtest[-1] == "-":  # partial words
                                    wordtest = wordtest[0:-1]
                                if "'" in wordtest:
                                    wordtest = wordtest.replace("'", "")
                                if len(wordPOSMap) > 0:
                                    found = False
                                    for n in range(len(wordPOSMap) - 1, -1, -1):
                                        if len(wordPOSMap[n][1]) > 0:
                                            j = wordPOSMap[n][1][-1] + 1
                                            # print j
                                            found = True
                                            break
                                    if not found:
                                        # if not possible go back to
                                        # the beginning!
                                        j = 0
                                else:
                                    j = 0
                                # print j
                            else:
                                mistranscribe = False
                                wordPOSMap.append((word, POSIndices))
                                errormessage = "WARNING: no/partial POS \
                                mapping for ''" + words[0] + "'' in file/utt:"\
                                    + str(utt.swda_filename) + "-" + \
                                    str(utt.transcript_index) + \
                                    "POSSIBLE COMMENT = " + str(possibleComment)
                                del words[0]  # remove word
                                if len(words) > 0:
                                    word = words[0]
                                    wordtest = re.sub(r"[\.\,\?\/\)\(\"\!\\]",
                                                      "", word)
                                    wordtest = wordtest.replace("(", "").\
                                        replace(")", "").replace("/", "")
                                # print errormessage
                                if errorLog:
                                    errorLog.write("possible wrong POS : " +
                                                   errormessage + "\n")
                                # raw_input()

                    # end of while loop (words)
                    if not len(wordPOSMap) == len(utt.text_words()):
                        print "Error "
                        print "Length mismatch in file/utt: " + \
                            str(utt.swda_filename) + str(utt.transcript_index)
                        print utt.text_words()
                        print wordPOSMap
                        raw_input()

                    wordPOSMapList.append(trans.conversation_no,
                                          str(utt.transcript_index),
                                          list(wordPOSMap))
                    # print "\nadded POSmap " + str(trans.swda_filename) + \
                    #"." + str(utt.transcript_index) + "\n"
                    csv_string = '"' + str(wordPOSMap) + '"'

                    corpus_file.write('"' + str(utt.conversation_no) +
                                      '"\t' + str(utt.transcript_index) +
                                      '\t' + csv_string + "\n")

                count += 1

        corpus_file.close()
        if errorLog:
            errorLog.close()


if __name__ == '__main__':
    t = TreeMapWriter()
