# Create the source part of the noisy channel through language modelling
# Consumes utterances or word streams word-by-word and outputs every single
# 'clean' underlying sequence of utterances
from __future__ import division
from copy import deepcopy
import math
import os

from deep_disfluency.language_model.ngram_language_model import NgramGraph
from deep_disfluency.language_model.ngram_language_model\
    import KneserNeySmoothingModel

_S_ = "<s>"  # lm start symbol
_E_ = "</s>"  # lm end symbol


def log(prob):
    if prob == 0.0:
        return - float("Inf")
    return math.log(prob, 2)


class SourceModel(object):
    def __init__(self, lm, pos_lm=None, uttseg=True):
        self.lm = lm  # language model
        self.pos_lm = pos_lm  # POS tag language model
        self.uttseg = uttseg  # whether this is for utterance segmentation too
        self.reset()

    def reset(self):
        self.word_graph = [_S_] * (self.lm.order - 1)  # the words
        self.pos_graph = [_S_] * (self.pos_lm.order - 1) if self.pos_lm \
            else None
        self.word_tree = []  # a list of dicts with pointers with the
        # tag, the probability of the sequence, and the mother node
        # (i.e. prevous word
        self.pos_tree = []

    def prune(self, method="log"):
        """Keeps the complexity linear after 9 words.
        Warning that this is in fact pruning backwards"""
        if method == "wml":
            def score(x):
                return log(0) if x[1][1] == 0 else x[1][1] / -x[1][2]
        elif method == "log":
            def score(x):
                return log(0) if x[1][1] == 0 else x[1][1]
        if len(self.word_tree) >= 9:
            n = int(len(self.word_tree[-1].keys()) / 3)
            n = 100
            top_n = sorted(self.word_tree[-1].items(),
                           key=lambda x: score(x),
                           reverse=True)[:n]
            # for k,v in top_n:
            #    print k,v
            self.word_tree[-1] = dict((k, v) for k, v in top_n)

    def consume_word(self, word, pos=None):
        """Adds the word and pos to current words and pos graph"""
        self.word_graph.append(word)  # always add to the top version
        # print self.word_graph
        if self.pos_lm and pos:
            self.pos_graph.append(pos)
        # if initial
        if len(self.word_tree) > 2:
            # self.prune()
            self.word_tree.append(dict())
            return  # TODO trying no more generation
        if len(self.word_tree) == 0:
            # init root of tree
            self.word_tree = [{0: ("0", -3.0, -1.0, 1, [_S_, _S_], 0)}]
            if self.pos_lm and pos:
                self.pos_tree = [{0: ("0", -3.0, -1.0, 1, [_S_, _S_], 0)}]
        new_word_dict = {}
        new_pos_dict = {}
        counter = 0
        language_models = [self.lm]
        if self.pos_lm and pos:
            language_models.append(self.pos_lm)
        new_prefix_dicts = [new_word_dict, new_pos_dict]
        trees = [self.word_tree, self.pos_tree]
        tags = ["<s/>", "<e/>", "<f/>"] if self.uttseg else ["<e/>", "<f/>"]
        for i in sorted(self.word_tree[-1].keys()):
            for fluency_tag in tags:
                context = self.word_tree[-1][i][4]
                if context[-1] == _S_ and fluency_tag == "<f/>":
                    # only two options on first branch
                    # TODO more problems here
                    continue
                # iterate over both modes
                for lm, prefix_dict, tree in zip(language_models,
                                                 new_prefix_dicts,
                                                 trees):
                    prob = tree[-1][i][1]  # the original log prob
                    unigram_prob = tree[-1][i][2]
                    if fluency_tag == "<e/>":
                        # no addition to the probability value keep trigram
                        # the same
                        trigram = [None] + tree[-1][i][4]
                    else:
                        if fluency_tag == "<s/>":
                            # calculate the prob of the last word ending
                            end_trigram = tree[-1][i][4] + [_E_]
                            prob += log(lm.ngram_prob(end_trigram, lm.order))
                            # TODO not sure if adding unigram needed here
                            trigram = [_S_, _S_] + [word]
                        else:
                            trigram = tree[-1][i][4] + [word]
                        prob += log(lm.ngram_prob(trigram, lm.order))
                        unigram = word
                        unigram_prob += log(lm.ngram_prob([unigram], 1))

                    # NB also add the decendent nodes for the mother node??
                    # add tuple
                    # fluency_tag, prob, unigram_prob, context, depth, parent_addr
                    prefix_dict[counter] = (fluency_tag,
                                            prob,
                                            unigram_prob,
                                            len(self.word_tree),
                                            trigram[1:], i)
                counter += 1

        self.word_tree.append(deepcopy(new_word_dict))
        if self.pos_lm and pos:
            self.pos_tree.append(deepcopy(new_pos_dict))

        # self.prune()

    def get_successor_node_value(self, parent_node_address,
                                 parent_node_value,
                                 fluency_tag, depth):
        """Compute what the nodes values should be from the
        parent node
        """
        context = parent_node_value[4]
        prob = parent_node_value[1]  # the original log prob
        unigram_prob = parent_node_value[2]
        # the 0-indexed position of word
        parent_word_index = parent_node_value[3] - 1 + (self.lm.order - 1)
        word = self.word_graph[parent_word_index + 1]
        if fluency_tag == "<e/>":
            # no addition to the probability value keep trigram
            # the same
            trigram = [None] + context
            # don't update probs or unigrams/contexts or..
            prob += log(0.01)
            unigram_prob += log(0.5)
        else:
            if fluency_tag == "<s/>":
                # calculate the prob of the last word ending
                end_trigram = context + [_E_]
                prob += log(self.lm.ngram_prob(end_trigram,
                                               self.lm.order))
                # TODO not sure if adding unigram needed here
                trigram = [_S_, _S_] + [word]
            else:
                trigram = context + [word]
            # word_index = len(self.word_graph) - 1
            prob += log(self.lm.ngram_prob(trigram, self.lm.order))
            unigram = word
            unigram_prob += log(self.lm.ngram_prob([unigram], 1))

        # NB also add the decendent nodes for the mother node??
        return (fluency_tag, prob, unigram_prob, depth, trigram[1:],
                parent_node_address)

    def find_or_generate_path_of_suffix_from_node(self, suffix, node_ID,
                                                  new=False, debug=False):
        """From a given node ID of tuple (tree depth, dict_id)
        (which must exist if this function is called),
        Find a path consistent with the suffix which matches
        the non-proper prefix of the words consumed so far.
        Returns the most probable path of nodes as a list.
        """
        # print "calling find from node", suffix, node_ID
        node_path = []  # returns the successors of the node, not itself
        assert len(suffix) <= (len(self.word_graph) - (self.lm.order - 1))
        depth, node_address = node_ID
        node_value = self.word_tree[depth].get(node_address)
        assert(node_value)
        if depth > (len(self.word_tree) - len(suffix) - 1):
            if debug:
                print "start node in front of suffix start, chain back", depth, len(suffix)
            # should be a fairly straightforward chain back
            last_node = (node_address, node_value)
            # print 'last node', last_node
            for d in range(
                    depth - 1, (len(self.word_tree) - len(suffix) - 1), -1):
                # get the father
                if debug:
                    print 'last node int', last_node
                    print d
                    print self.word_tree[d]
                last_node = filter(lambda x: x[0] == last_node[1][-1],
                                   self.word_tree[d].items())[0]
                depth = d
            node_value = last_node[1]
            node_address = last_node[0]
        node_path = [(node_address, node_value)]
        # now chain through the successor depths to find a path
        # through consistent with the suffix tags, else
        # create the path
        for d, tag in zip(range(depth + 1, len(self.word_tree)), suffix):
            if not new:
                # 1. see if successor node exists with right tag
                nodes = filter(lambda x: x[1][0] == tag,
                               self.word_tree[d].items())
                nodes = filter(lambda x: x[1][-1] == node_address,
                               nodes)
                if nodes:
                    # get the new successor nodes
                    node_address, node_value = nodes[0]
                    node_path.append((node_address, node_value))
                    continue
                # if not make one
                else:
                    new = True
            # make the new node at the current level

            # do the new node value calculations from the current node value
            node_value = self.get_successor_node_value(node_address,
                                                       node_value, tag, d)

            node_address = 0 if len(self.word_tree[d].items()) == 0 \
                else max(self.word_tree[d].items(),
                         key=lambda x: x[0])[0] + 1
            self.word_tree[d][node_address] = node_value
            node_path.append((node_address, node_value))
        return node_path

    def find_or_generate_best_path_of_suffix(self, suffix):
        """Find the node values relevant for the suffix if possible,
        else generate the best path from an anchor node
        using find_or_generate_path_of_suffix_from_node method
        """
        node_path = []
        for s, d in zip(
                range(0, len(suffix)),
                range(len(self.word_tree) - len(suffix),
                      len(self.word_tree))):
            # print "s,d", s,d
            # if we're in the loop we're still assuming there is
            # an existing path
            tag = suffix[s]
            if s == 0:
                # first one, just get the ones with the right tags
                nodes = filter(lambda x: x[1][0] == tag,
                               self.word_tree[d].items())
            else:
                # otherwise, get the ones with the right tags which
                # are the children of the existing nodes
                successors = filter(lambda x: x[1][0] == tag,
                                    self.word_tree[d].items())
                nodes = filter(lambda x: any([x[1][-1] ==
                                              n[0] for n in nodes]),
                               successors)
            if nodes:
                node_path.append(deepcopy(nodes))
            else:
                # get most likely node at d-1
                if node_path:
                    # first flatten this to the most likely
                    # and chain back
                    last_node = max(node_path[-1],
                                    key=lambda x: x[1][1])
                    new_node_path = [last_node]
                    for b in range(len(node_path) - 2, -1, -1):
                        # get the father
                        last_node = filter(lambda x: x[0] == last_node[1][-1],
                                           node_path[b])[0]
                        assert(last_node)
                        new_node_path = [last_node] + new_node_path
                    node_path = new_node_path
                    node = node_path[-1]
                else:
                    # empty node path so far
                    # just get most likely node at d-1
                    node = max(self.word_tree[d - 1].items(),
                               key=lambda x: x[1][1])
                node_ID = (d - 1, node[0])
                path_tail = self.find_or_generate_path_of_suffix_from_node(
                    suffix[s:],
                    node_ID,
                    new=True)
                return node_path + path_tail
        # got this far we have found a path through of existing nodes
        # print "got to end"
        last_node = max(node_path[-1], key=lambda x: x[1][1])
        new_node_path = [last_node]
        # print 'last node', last_node
        for b in range(len(node_path) - 2, -1, -1):
            # get the father
            # print 'last node int', last_node
            # print "node path[b]", node_path[b]
            last_node = filter(lambda x: x[0] == last_node[1][-1],
                               node_path[b])[0]
            assert(last_node)
            new_node_path = [last_node] + new_node_path
        node_path = new_node_path
        return node_path

    def get_log_diff_of_tag_suffix(self, suffix, n=1, start_node_ID=None,
                                   pos=True):
        """For a given suffix of edit operations on the original
        string, give the probability of that string in the lm
        in terms of the gain of negative log prob from
        n words back from the end of the suffix"""
        assert len(suffix) <= len(self.word_tree)
        # print "suffix", suffix
        if not suffix:
            print "WARNING empty suffix queried"
            return 0, None
        if len(self.word_tree) == 2 and suffix[0] == "<f/>":  # i.e. first word
            return log(0.0), None  # the only illegal sequence
        if start_node_ID:
            # we have a starting point on the word_tree graph we know
            # the suffix is consistent with up to that point
            # (this saves time)
            path = self.find_or_generate_path_of_suffix_from_node(
                suffix, start_node_ID)
            #start_node = self.word_tree[start_node_ID[0]][start_node_ID[1]]
            #orig_log_prob = start_node[1]
            #orig_unigram_log_prob = start_node[2]
        else:
            path = self.find_or_generate_best_path_of_suffix(suffix)
        orig_log_prob = path[min([len(path) - 1, len(path) - (1 + n)])][1][1]
        orig_unigram_log_prob = path[min(
            [len(path) - 1, len(path) - (1 + n)])][1][2]
        final_log_prob = path[-1][1][1]
        diff = final_log_prob - orig_log_prob
        # TODO what if diff is 0?
        # print 'path'
        # for n in path:
        #    print n
        # print '...'
        final_unigram_log_prob = path[-1][1][2]
        unigram_diff = final_unigram_log_prob - orig_unigram_log_prob
        wml_diff = 0.999 if unigram_diff == 0 else 1 - \
            (((diff / - unigram_diff)) / -3.5)
        # print wml_diff
        #diff = log(wml_diff)
        diff = log(0.01) if diff == 0 else diff  # too much weight for 0s
        return diff, (path[-1][1][3], path[-1][0])

    def get_top_n_sequences(self, n):
        """Get the most probable n sequences.
        """
        top_n = []
        lm_tree_dict = self.word_tree[-1]
        final_nodes = sorted(lm_tree_dict.items(),
                             key=lambda x: log(0) if x[1][2] == 0
                             else x[1][1] / -x[1][2],
                             reverse=True)[:n]
        for f in final_nodes:
            # each of the top ones get the sequences
            father_node = f[1][-1]
            tag = f[1][0]
            sequence = [tag]
            for i in range(len(self.word_tree) - 2, -1, -1):
                lm_tree_dict = self.word_tree[i]
                node_val = lm_tree_dict[father_node]
                tag = node_val[0]
                sequence = [tag] + sequence
                father_node = node_val[-1]
            top_n.append(deepcopy(sequence))
        for seq, x in zip(top_n, final_nodes):
            wml = log(0) if x[1][2] == 0 else x[1][1] / -x[1][2]
            print seq, wml
        return top_n


class LMTester(object):

    def __init__(self):
        self.init_language_models(None, None, None)

    def init_language_models(self, language_model=None,
                             pos_language_model=None,
                             edit_language_model=None):
        print "Init language models ..."
        pos = True
        clean_model_dir = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/lm_corpora"
        if language_model:
            self.lm = language_model
        else:
            print "No language model specified, using default switchboard one"
            lm_corpus_file = open(clean_model_dir +
                                  "/swbd_disf_train_1_clean.text")
            lines = [line.strip("\n").split(",")[1] for line in lm_corpus_file
                     if "POS," not in line and not line.strip("\n") == ""]
            split = int(0.9 * len(lines))
            lm_corpus = "\n".join(lines[:split])
            heldout_lm_corpus = "\n".join(lines[split:])
            lm_corpus_file.close()
            self.lm = KneserNeySmoothingModel(
                order=3,
                discount=0.7,
                partial_words=True,
                train_corpus=lm_corpus,
                heldout_corpus=heldout_lm_corpus,
                second_corpus=None)
        if pos_language_model:
            self.pos_lm = pos_language_model
        elif pos:
            print "No pos language model specified, \
            using default switchboard one"
            lm_corpus_file = open(clean_model_dir +
                                  "/swbd_disf_train_1_clean.text")
            lines = [line.strip("\n").split(",")[1] for line in lm_corpus_file
                     if "POS," in line and not line.strip("\n") == ""]
            split = int(0.9 * len(lines))
            lm_corpus = "\n".join(lines[:split])
            heldout_lm_corpus = "\n".join(lines[split:])
            lm_corpus_file.close()
            self.pos_lm = KneserNeySmoothingModel(
                order=3,
                discount=0.7,
                partial_words=True,
                train_corpus=lm_corpus,
                heldout_corpus=heldout_lm_corpus,
                second_corpus=None)


if __name__ == '__main__':
    lmtest = LMTester()
    s = SourceModel(lmtest.lm, lmtest.pos_lm)
    words = "you just really cant cant tell whats going to happen yeah".split()
    pos = "PRP RB RB MDRB MDRB VB WPBES VBG TO VB UH".split()
    tags = ['<s/>', '<e/>', '<e/>', '<e/>', '<f/>', '<f/>', '<f/>', '<f/>',
            '<f/>', '<s/>', '<s/>']
    for w, p, i in zip(words, pos, range(0, len(words))):
        s.consume_word(w)
        print "dimen of word tree,", len(s.word_tree), len(s.word_tree[-1])
        # print s.word_tree[-1]
        # top_n = s.get_top_n_sequences(5)
        # print top_n
        # print top_n[:10]
        # raw_input()
    # check all
        print s.get_log_diff_of_tag_suffix(tags[:i], n=1)

    print s.get_log_diff_of_tag_suffix([
        "<s/>", "<s/>", "<s/>"], n=1, start_node_ID=(10, 0))
