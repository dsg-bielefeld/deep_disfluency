# -*- coding: utf-8 -*-
from collections import defaultdict
import re
import string
import csv
import os
from glob import iglob
from nltk import tree

punctuation_pos = [".", ",", ":", ";", "!", "PUL", "PUN", "PUQ", "PUR"]


class TreeMap(list):
    """A list object which contains tuples which map from the index of
    words in an utterance to a tree number in a treebank and the
    node(s) the word relates to.
    """

    def __init__(self, args, treenumbers):
        list.__init__(self, args)
        self.treebank_numbers = []
        self.transcript_numbers = []
        for tn in treenumbers:
            self.treebank_numbers.append(tn[0])
            self.transcript_numbers.append([tn[1], tn[2]])

    def get_wordtreemap(self, word_index):
        """Return the word>treenode map for a word at a word index."""
        try:
            return self[word_index]
        except IndexError:
            return None

    def get_trees(self, trans):
        return [trans.utterances[index[0]].trees[index[1]]
                for index in self.transcript_numbers]

    def treenumbers(self, trans):
        trees = self.get_trees(trans)
        return [tree[1] for tree in trees]

    def get_POS(self, trans, utt):
        """Return the POS for each word in the map where there is one,
        else returns a null token string.
        """
        POS = []
        previous = None
        trees = self.get_trees(trans)
        for wordMap in self:
            if wordMap[1] != [] and len(wordMap[1]) >= 1:
                try:
                    if not wordMap[1][0][0] == previous:
                        mytree = trees[wordMap[1][0][0]]
                        mypos = mytree.pos()
                        previous = wordMap[1][0][0]
                except IndexError:
                    print "ERROR can't get tree from index"
                    print "TreeMap.get_POS(utt)"
                    print utt.swda_filename
                    print utt.transcript_index
                    raw_input()
                pos = ""
                # concatenating now
                for n in wordMap[1]:
                    if not mypos[n[1]][1] in punctuation_pos:
                        # print mypos[n[1]][1]
                        pos += mypos[n[1]][1]
                POS.append(pos)
            else:
                POS.append("null")
        if not len(POS) == len(utt.text_words()):
            print "ERROR: uneven lengths: TreeMap.get_POS"
            print utt.swda_filename
            print utt.transcript_index
            raw_input()
        return POS

    def get_last_TreeNumber(self):
        """Return last mapped word in the list's tree number"""
        for i in range(len(self) - 1, -1, -1):
            pair = self[i]
            if len(pair[1]) > 0:
                return pair[1][-1][0]
        print "WARNING: no trees mapped to in this utt!"
        return None

    def get_first_TreeNumber(self):
        """Return first mapped word in the list's tree number"""
        for i in range(len(self)):
            pair = self[i]
            if len(pair[1]) > 0:
                return pair[1][0][0]
        print "WARNING: no trees mapped to in this utt!"
        return None

    def get_ancestor_nodes(self, mytree, nodeposition):
        """Return top-down path of ancestors of given node."""
        if nodeposition < 0:
            raise IndexError('Leaf number must be non-negative')
        if nodeposition == ():
            return []  # could be root

        if isinstance(nodeposition, int):
            nodeposition = mytree.leaf_treeposition(nodeposition)
        mothers = []
        # start with entire tree, and root node position ()
        # stack should only contain candidate ancestors or the node itself
        stack = [(mytree, ())]

        while stack:
            value, treepos = stack.pop()

            if treepos == ():  # if root node..
                mothers.append(value.label())

            # keep going till we reach our target node
            if treepos == nodeposition:
                return mothers

            for i in range(len(value) - 1, -1, -1):
                if ((isinstance(value[i], tree.Tree) and
                     value[i].height() < mytree.height() + 1) or
                        (isinstance(value[i], str))):
                    # check to see if the value is in the path
                    # should all be the same level from top?
                    if(self.ancestor_of(treepos + (i,), nodeposition)):
                        # just append candidate mother(s)
                        # should only be one in a tree
                        stack.append((value[i], treepos + (i,)))
                        mothers.append(value[i].label())
                        break  # assuming single mother should stop here
                    elif treepos + (i,) == nodeposition:
                        stack.append((value[i], treepos + (i,)))
                        break

    def get_ancestor_node_addresses(self, mytree, nodeposition):
        """Return top-down path of pairs of ancestor node addresses
        and their value for given node at nodeposition.
        """
        if nodeposition < 0:
            raise IndexError('leafnumber must be non-negative')
        if nodeposition == ():
            return []  # could be root

        mothers = []
        # start with entire tree, and root node position ()
        # stack should only contain candidate ancestor or the node itself
        stack = [(mytree, ())]

        while stack:
            value, treepos = stack.pop()

            if treepos == ():  # if root node..
                mothers.append((value.label(), treepos))
            # keep going till we reach our target node
            if treepos == nodeposition:
                break

            for i in range(len(value) - 1, -1, -1):
                if ((isinstance(value[i], tree.Tree) and
                     value[i].height() < mytree.height() + 1) or
                        (isinstance(value[i], str))):
                    # check to see if the value is in the path
                    if self.ancestor_of(treepos + (i,), nodeposition):
                        # just append candidate mother(s)
                        # should only be one in a tree
                        stack.append((value[i], treepos + (i,)))
                        mothers.append((value[i].label(), treepos + (i,)))
                        break  # assuming single mother stop here
                    elif treepos + (i,) == nodeposition:
                        stack.append((value[i], treepos + (i,)))
                        break
        return mothers

    def get_word_tree_depths(self, transcript_index, mytree, myTreeNumber):
        """Return a list of tree depths for each leaf in tree mapped
        to in the Treemap in a given tree (not just for each word due
        to "it's" type cases.
        """
        treeDepths = []
        depth = 0
        for n in range(len(self)):
            # only for words with mapping(s) and only for this tree number
            if self[n][1] == []:
                continue
            if self[n][1][0][0] != myTreeNumber:
                continue
            # we might have multiple nodes for each word here,
            # iterate through mappings
            for y in range(len(self[n][1])):
                # get leaf number of the yth nodeposition,
                # usually only one per node!
                depth = len(mytree.leaf_treeposition(self[n][1][y][1]))
                myMothers = self.get_ancestor_nodes(
                    mytree,
                    mytree.leaf_treeposition(self[n][1][y][1]))
                if "EDITED" in myMothers:
                    # in reparandum
                    newpath = []
                    # create new path with edited nodes removed appropriately
                    for int1 in range(len(myMothers) - 1, -1, -1):
                        if myMothers[int1] == "EDITED":  # ignore edited nodes
                            pass
                        elif (int1 < (len(myMothers) - 1) and
                              len(newpath) > 0
                              and myMothers[int1 + 1] == "EDITED"):
                            starts = [match.start() for match in
                                      re.finditer(re.escape(myMothers[int1]),
                                                  newpath[0])]
                            # check to see if unfinished node
                            if (myMothers[int1] == newpath[0] or
                                (len(starts) > 0 and starts[0] == 0
                                 and newpath[0][-3:] == "UNF")):
                                pass  # i.e. do not add
                            else:
                                # put to front of list
                                newpath.insert(0, myMothers[int1])
                        else:
                            newpath.insert(0, myMothers[int1])

                    depth = len(newpath)
                # append (word index,category,depth) triple
                treeDepths.append(
                    [[transcript_index, n], myMothers[-1], depth])
        return treeDepths

    def get_word_tree_path_lengths(self, transcript_index, mytree,
                                   myTreeNumber):
        """Return a list of the backwards treepath distance between
        each leaf in tree mapped to in the Treemap in a given tree.
        Should be for each word as each transcribed word in TreeMap
        that has a mapping mapped to (possibly multiple) leaves
        """

        motherList = []  # top down list of all mother addresses for each node
        treePathLengths = []
        for n in range(len(self)):
            # only for words with mapping(s) and only for this tree number
            if (self[n][1] == [] or len(self[n][1]) == 0):
                continue
            if self[n][1][0][0] != myTreeNumber:
                continue
            # we might have multiple nodes for each word here
            # iterate through mappings
            # Get all mother addresses, minus the edited nodes,
            # and add to motherList
            for y in range(len(self[n][1])):
                myMothers = self.get_ancestor_node_addresses(
                    mytree,
                    mytree.leaf_treeposition(self[n][1][y][1]))

                motherList.append(myMothers)
                if len(motherList) > 1:
                    # look for lowest common mother and get path:
                    found = False
                    penalty = 0  # reduces final length (edited bits)
                    myedited = []
                    previousedited = []
                    for i in range(len(myMothers) - 1, -1, -1):
                        if myMothers[i][0] == "EDITED":
                            if (i > 0 and not myMothers[i] in myedited):
                                myedited.append(myMothers[i])
                                penalty = penalty - 1
                                # check for resolution of the UNF node
                                if (myMothers[i - 1][0] != "EDITED"
                                        and not myMothers[i - 1] in myedited):
                                    # forwards pass through other nodes until
                                    # non-edit is reached and comparison made
                                    for k in range(i + 1, len(myMothers)):
                                        if (myMothers[k][0] == "EDITED"
                                                and not myMothers[k] in myedited):
                                            myedited.append(myMothers[k])
                                            penalty = penalty - 1
                                        elif (myMothers[k][0] != "EDITED"
                                              and not myMothers[k] in myedited
                                              and myMothers[i - 1][0] ==
                                              myMothers[k][0][:len(myMothers[i - 1][0])]):
                                            myedited.append(myMothers[k])
                                            penalty = penalty - 1
                        # scan through the other one
                        for j in range(len(motherList[-2]) - 1, -1, -1):
                            if (j > 0 and motherList[-2][j][0] == "EDITED"
                                and not motherList[-2][j] in previousedited
                                and len(motherList[-2][j][1]) >
                                    len(myMothers[i][1])):
                                penalty = penalty - 1
                                previousedited.append(motherList[-2][j])
                                if (motherList[-2][j - 1][0] != "EDITED"
                                    and not motherList[-2][j - 1]
                                        in previousedited):
                                    for k in range(j + 1, len(motherList[-2])):
                                        if (motherList[-2][k][0] == "EDITED"
                                            and not motherList[-2][k]
                                                in previousedited):
                                            m = motherList[-2][k]
                                            previousedited.append(m)
                                            penalty = penalty - 1
                                        elif (motherList[-2][k][0] != "EDITED"
                                              and not motherList[-2][k]
                                              in previousedited and
                                              motherList[-2][j - 1][0] ==
                                              motherList[-2][k][0][:len(motherList[-2][j - 1][0])]):
                                            m = motherList[-2][k]
                                            previousedited.append(m)
                                            penalty = penalty - 1
                            if myMothers[i][1] == motherList[-2][j][1]:
                                if myMothers[i][0] == "EDITED":
                                    continue  # look for next one up!
                                commonMother = myMothers[i][0]
                                pathlength = (len(myMothers) - i) +\
                                    (len(motherList[-2]) - j) + penalty
                                found = True
                                break
                        if found:
                            break

                else:
                    pathlength = "<S>"
                    commonMother = "NULL"
                # append word no/POS/pathlength/commonMother node tuple
                treePathLengths.append(((transcript_index, n),
                                        myMothers[-1],
                                        pathlength,
                                        commonMother))
        return treePathLengths

    def get_edit_words(self, transcript_index, trees):
        editwords = []
        for n in range(len(self)):
            if (self[n][1] == [] or len(self[n][1]) == 0):
                editwords.append(False)
                continue
            myTreeNumber = self[n][1][0][0]
            mytree = trees[myTreeNumber]

            found = False
            # going through multiple possible mappings
            for y in range(len(self[n][1])):
                if found:
                    break
                myMothers = self.get_ancestor_node_addresses(
                    mytree,
                    mytree.leaf_treeposition(self[n][1][y][1]))

                if len(myMothers) > 1:
                    for i in range(len(myMothers) - 1, -1, -1):
                        if myMothers[i][0] == "EDITED":
                            editwords.append(True)
                            found = True
                            break
            if not found:
                editwords.append(False)

        return editwords

    def get_edited_word_positions(self, trans, index):
        """Return a list of booleans where TRUE == edited word"""
        trees = self.get_trees(trans)
        edited = self.get_edit_words(index, trees)
        assert len(trans.utterances[index].text_words())\
            == len(edited), trans.utterances[index].text_words()
        return edited

    def get_path_lengths(self, trans, index):
        trees = self.get_trees(trans)
        overall = []
        treepaths = defaultdict(int)
        i = 0
        for tree in trees:
            newlist = self.get_word_tree_path_lengths(index, tree, i)
            overall.extend(newlist)
            i += 1
        for o in overall:
            treepaths[o[0][1]] = o[2]
        return treepaths

    def ancestor_of(self, motherpos, daughterpos):
        """Boolean as to whether first argument node is an ancestor of
        the second.
        """
        if len(motherpos) >= len(daughterpos):
            return False
        for n in range(0, len(motherpos)):
            if not motherpos[n] == daughterpos[n]:
                return False
        return True


class TreeMapCorpus(dict):

    def __init__(self, readIn, errorLog, mapdir="swda_tree_pos_maps", *args):
        dict.__init__(self, args)
        self.errorlog = errorLog
        if readIn:
            print "loading treemap files..."
            for filename in sorted(iglob(os.path.join(mapdir, "Tree_map*"))):
                self.iter_treemaps_from_file(filename)
            print "treemaplist length: " + str(len(self))

    def get_treemap(self, trans, utt):
        if utt is None:
            # print "WARNING null utt from treemap"
            return None
        map_key = str(utt.conversation_no) + ":" + str(utt.transcript_index)
        try:
            treemap = self[map_key]
            if len(treemap) != len(utt.text_words()):
                warning = "ERROR: TREEMAP HERE NOT IN LINE WITH UTT" +\
                    utt.swda_filename + str(utt.transcript_index) + "\n"
                print warning
                print utt.text_words()
                print treemap
                if not self.errorlog is None:
                    self.errorlog.write(warning)
                raw_input()
                return None
            return treemap
        except BaseException:
            if not len(utt.trees) == 0 and not utt.damsl_act_tag() == "x":
                warning = "WARNING NO TREE MAP FOR" + utt.swda_filename +\
                    str(utt.transcript_index) + " from map key" +\
                    str(map_key) + "\n"
                print warning
                if not self.errorlog is None:
                    self.errorlog.write(warning)
            return None

    def append(self, transFilename, uttTransNumber, treeNumbers, wordTreeMap):
        key = transFilename + ":" + str(uttTransNumber)
        self[key] = TreeMap(wordTreeMap, treeNumbers)

    def parse_list(self, string, length, bracketstart='[', bracketend=']'):
        """Returns list object from string representation.
        e.g. returns list [1,2] from string "[1,2]"
        """
        chars = string
        current = ""
        mylist = []
        for char in chars:
            if char == " ":
                continue
            if char == bracketstart:
                pass
            elif char == ",":
                mylist.append(int(current))
                current = ""
            elif char == bracketend:
                mylist.append(int(current))
                break
            else:
                current += char
        assert(len(mylist) == length)
        return mylist

    def iter_treemaps_from_file(self, filename):
        """ Iterate through the csv file of the treemaps, creating a
        TreeMap object for each one.
        """
        myreader = csv.reader(open(filename, 'r'), delimiter='\t',
                              quotechar=None)
        myrows = list(myreader)
        for row in myrows:
            wordBool = False
            mappings = False
            word = ''
            mymappings = []
            myTreemap = []
            i = 0  # character pointer
            j = 0  # word pointer

            while i < len(row[2]):  # goes through character by character
                string = row[2][i]
                if (i < len(row[2]) - 1 and string == '(' and
                        mappings is False and not row[2][i + 1] == '('
                        and not row[2][i - 1] == '('):
                    wordBool = True
                    if (row[2][i + 1] == "'" or row[2][i + 1] == '"'):
                        i = i + 1

                elif (i < len(row[2]) - 3 and row[2][i + 1] == ',' and
                      row[2][i + 3] == '[' and wordBool):
                    wordBool = False
                    mappings = True
                    if (string == "'" or string == '"'):
                        i = i + 2
                    else:
                        i += 1

                elif (string == '(' and mappings):
                    i += 1
                    treeNum = row[2][i]
                    while (not row[2][i] == ","):
                        i += 1
                    treeNum = int(treeNum)
                    i += 1
                    nodeNum = row[2][i]
                    i += 1
                    while not row[2][i] == ")":
                        nodeNum = nodeNum + row[2][i]
                        i += 1
                    nodeNum = int(nodeNum)
                    mymappings.append((treeNum, nodeNum))

                elif (string == ']' and mappings):
                    mappings = False

                elif wordBool is True:
                    word += string

                elif (not wordBool and not mappings) and \
                    (word != '' and string == ')' and not
                     row[2][i + 1] == ')' and not row[2][i - 1] == '('):
                    myTreemap.append([word, list(mymappings)])
                    word = ''  # reset word
                    mymappings = []  # reset mapping
                    j += 1
                i += 1
            treeNumbers = []
            tn = row[3].replace('"', "")
            tn = re.split(";", tn)
            for mystring in tn:
                treenums = self.parse_list(mystring, 3)
                treeNumbers.append(list(treenums))  # length of each list
            file_ref = row[0].replace('"', "")
            filestring = file_ref + ":" + str(int(row[1]))
            self[filestring] = TreeMap(myTreemap, treeNumbers)

    def iter_treemaps(self):
        for t in self.keys():
            yield self[t]


class POSMap(list):
    """A list object which contains tuples which map from the index of
    words in an utterance to a the POS tag in a list of POS tags
    which the word relates to.
    """

    def __init__(self, *args):
        list.__init__(self, *args)

    def get_POS(self, utt):
        """Return the POS for each word in the map,
        else return a null token string.
        Much faster than TreeMap.get_POS(utt)
        """
        POS = []
        mypos = utt.regularize_pos_lemmas()
        for wordMap in self:
            if wordMap[1] != [] and len(wordMap[1]) >= 1:
                try:
                    pos = ""
                    for n in wordMap[1]:  # CONCATENATING
                        if not mypos[n][1] in punctuation_pos:
                            pos += mypos[n][1]
                    POS.append(pos)
                except IndexError:
                    print "can't get POS from index: POSMap.get_POS(utt)"
                    print utt.swda_filename
                    print utt.transcript_index
                    raw_input()
            else:
                POS.append("null")
        if not len(POS) == len(utt.text_words()):
            print "uneven lengths: POSMap.get_POS mytree.py"
            print utt.swda_filename
            print utt.transcript_index
            raw_input()
        return POS


class POSMapCorpus(dict):

    def __init__(self, readIn, errorLog, mapdir="swda_tree_pos_maps", *args):
        dict.__init__(self, args)
        self.errorlog = errorLog
        if readIn:
            print "loading posmap files..."
            for filename in sorted(iglob(os.path.join(mapdir, "POS_map*"))):
                self.iter_POSmaps_from_file(filename)
            print "posmaplist length: " + str(len(self))

    def get_POSmap(self, trans, utt):
        if utt is None:
            # print "null utt"
            return None
        map_key = str(utt.conversation_no) + ":" + str(utt.transcript_index)
        if not self.get(map_key) is None:
            posmap = self[map_key]
            if len(posmap) != len(utt.text_words()):
                warning = "WARNING: posMAP HERE NOT IN LINE WITH UTT" +\
                    str(utt.swda_filename) + str(utt.transcript_index) + "\n"
                print warning
                print utt.text_words()
                print posmap
                print sorted(self.keys())
                if not self.errorlog is None:
                    self.errorlog.write(warning)
                raw_input()
                return None
            return posmap
        else:
            if not len(utt.pos) == 0 and not utt.pos[0] == ".//.":
                warning = "WARNING NO POS FOR " + str(utt.conversation_no) +\
                    "-" + str(utt.transcript_index) + "\n"
                print warning
                print utt.text_words()
                print sorted(self.keys())[:20]
                print "map key", map_key
                if not self.errorlog is None:
                    self.errorlog.write(warning)
                raw_input()
            return None

    def append(self, transFilename, uttTransNumber, wordPOSMap):
        self[transFilename + ":" + str(uttTransNumber)] = wordPOSMap

    def iter_POSmaps_from_file(self, filename):
        """ Iterate through the csv file of the POSmaps, creating
        generating a list of
        (triple of filename, utterance index and POSmap)
        """
        myreader = csv.reader(open(filename, 'r'), delimiter='\t',
                              quotechar=None)
        myrows = list(myreader)
        for row in myrows:

            wordBool = False
            mappings = False
            word = ''
            mymappings = []
            myPOSmap = []
            i = 0  # character pointer
            j = 0  # word pointer

            while i < len(row[2]):  # goes character by character!
                string = row[2][i]
                if (i < len(row[2]) - 1 and string == '(' and not mappings
                        and not row[2][i + 1] == '(' and not row[2][i - 1] == '('):
                    wordBool = True
                    if (row[2][i + 1] == "'" or row[2][i + 1] == '"'):  # skip
                        i = i + 1
                elif (i < len(row[2]) - 3 and row[2][i + 1] == ','
                      and row[2][i + 3] == '[' and wordBool):
                    wordBool = False
                    mappings = True
                    if (string == "'" or string == '"'):
                        i = i + 2
                    else:
                        i += 1
                elif mappings and not string in [",", "[", "]"]:
                    # i.e. we've got a mapping
                    # i += 1 # move on to
                    posNum = str(row[2][i])  # POS position number.
                    i += 1  # move on one
                    while (not row[2][i] in [",", "]"]):
                        posNum = posNum + row[2][i]
                        i += 1
                    posNum = int(posNum)
                    i -= 1  # purely convention as will happen below anyway
                    mymappings.append(posNum)
                elif (string == ']' and mappings):
                    mappings = False  # might return empty list
                elif wordBool:
                    word += string
                elif (not wordBool and not mappings and word != ''
                      and string == ')' and not row[2][i + 1] == ')'
                      and not row[2][i - 1] == '('):
                    myPOSmap.append([word, list(mymappings)])
                    word = ''  # reset word
                    mymappings = []  # reset mapping
                    j += 1
                i += 1
            file_ref = row[0].replace('"', "")
            filestring = file_ref + ":" + str(int(row[1]))
            # print filestring
            self[filestring] = POSMap(myPOSmap)  # POSmaps

    def iter_POSmaps(self):
        for t in self.keys():
            yield self[t]
