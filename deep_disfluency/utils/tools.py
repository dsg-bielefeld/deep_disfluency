import random
import numpy as np
import itertools
import re
from collections import defaultdict
import os


def get_tags(s, open_delim='<', close_delim='/>'):
    """Iterator to spit out the xml style disfluency tags in a given string.

    Keyword arguments:
    s -- input string
    """
    while True:
        # Search for the next two delimiters in the source text
        start = s.find(open_delim)
        end = s.find(close_delim)
        # We found a non-empty match
        if -1 < start < end:
            # Skip the length of the open delimiter
            start += len(open_delim)
            # Spit out the tag
            yield open_delim + s[start:end].strip() + close_delim
            # Truncate string to start from last match
            s = s[end + len(close_delim):]
        else:
            return


def remove_uttseg_tag(tag):
    tags = get_tags(tag)
    final_tag = ""
    for t in tags:
        m = re.search(r'<[ct]*/>', t)
        if m:
            continue
        final_tag += t
    return final_tag


def convert_to_simple_label(tag, rep="disf1_uttseg"):
    """Takes the complex tag set and gives back the simple,
    smaller version with ten tags:
    """
    disftag = "<f/>"
    if "<rm-" in tag:
        disftag = "<rm-0/>"
    elif "<e" in tag:
        disftag = "<e/>"
    if "uttseg" in rep:  # if combined task with TTO
        m = re.search(r'<[ct]*/>', tag)
        if m:
            return disftag + m.group(0)
        else:
            print "WARNING NO TAG", tag
            return ""
    return disftag  # if not TT0


def convert_to_simple_idx(tag, rep='1_trp'):
    tag = convert_to_simple_label(tag, rep)
    simple_tags = """<e/><cc/>
    <e/><ct/>
    <e/><tc/>
    <e/><tt/>
    <f/><cc/>
    <f/><ct/>
    <f/><tc/>
    <f/><tt/>
    <rm-0/><cc/>
    <rm-0/><ct/>""".split("\n")
    simple_tag_dict = {}
    for s in range(0, len(simple_tags)):
        simple_tag_dict[simple_tags[s].strip()] = s
    return simple_tag_dict[tag]


def convert_from_full_tag_set_to_idx(tag, rep, idx_to_label):
    """Maps from the full tag set of trp repairs to the new dictionary"""
    if "simple" in rep:
        tag = convert_to_simple_label(tag)
    for k, v in idx_to_label.items():
        if v in tag:  # a substring relation
            return k


def add_word_continuation_tags(tags):
    """In place, add a continutation tag to each word:
    <cc/> -word continues current dialogue act and the next word will also
           continue it
    <ct/> -word continues current dialogue act and is the last word of it
    <tc/> -word starts this dialogue act tag and the next word continues it
    <tt/> -word starts and ends dialogue act (single word dialogue act)
    """
    tags = list(tags)
    for i in range(0, len(tags)):
        if i == 0:
            tags[i] = tags[i] + "<t"
        else:
            tags[i] = tags[i] + "<c"
        if i == len(tags) - 1:
            tags[i] = tags[i] + "t/>"
        else:
            tags[i] = tags[i] + "c/>"
    return tags


def verify_disfluency_tags(tags, normalize_ID=False):
    """Check that the repair tags sequence is valid.

    Keyword arguments:
    normalize_ID -- boolean, whether to convert the repair ID
    numbers to be derivable from their unique RPS position in the utterance.
    """
    id_map = dict()  # map between old ID and new ID
    # in first pass get old and new IDs
    for i in range(0, len(tags)):
        rps = re.findall("<rps id\=\"[0-9]+\"\/>", tags[i])
        if rps:
            id_map[rps[0][rps[0].find("=") + 2:-3]] = str(i)
    # key: old repair ID, value, list [reparandum,interregnum,repair]
    # all True when repair is all there
    repairs = defaultdict(list)
    for r in id_map.keys():
        repairs[r] = [None, None, None]  # three valued None<False<True
    print repairs
    # second pass verify the validity of the tags
    # and (optionally) modify the IDs
    for i in range(0, len(tags)):  # iterate over all tag strings
        new_tags = []
        if tags[i] == "":
            assert(all([repairs[ID][2] or
                        repairs[ID] == [None, None, None]
                        for ID in repairs.keys()])),\
                "Unresolved repairs at fluent tag\n\t" + str(repairs)
        for tag in get_tags(tags[i]):  # iterate over all tags
            print i, tag
            if tag == "<e/>":
                new_tags.append(tag)
                continue
            ID = tag[tag.find("=") + 2:-3]
            if "<rms" in tag:
                assert repairs[ID][0] is None,\
                    "reparandum started parsed more than once " + ID
                assert repairs[ID][1] is None,\
                    "reparandum start again during interregnum phase " + ID
                assert repairs[ID][2] is None,\
                    "reparandum start again during repair phase " + ID
                repairs[ID][0] = False  # set in progress
            elif "<rm " in tag:
                assert repairs[ID][0] is not None,\
                    "mid reparandum tag before reparandum start " + ID
                assert repairs[ID][2] is None,\
                    "mid reparandum tag in a interregnum phase or beyond " + ID
                assert repairs[ID][2] is None,\
                    "mid reparandum tag in a repair phase or beyond " + ID
            elif "<i" in tag:
                assert repairs[ID][0] is not None,\
                    "interregnum start before reparandum start " + ID
                assert repairs[ID][2] is None,\
                    "interregnum in a repair phase " + ID
                if repairs[ID][1] is None:  # interregnum not reached yet
                    repairs[ID][0] = True  # reparandum completed
                repairs[ID][1] = False  # interregnum in progress
            elif "<rps" in tag:
                assert repairs[ID][0] is not None,\
                    "repair start before reparandum start " + ID
                assert repairs[ID][1] != True,\
                    "interregnum over before repair start " + ID
                assert repairs[ID][2] is None,\
                    "repair start parsed twice " + ID
                repairs[ID][0] = True  # reparanudm complete
                repairs[ID][1] = True  # interregnum complete
                repairs[ID][2] = False  # repair in progress
            elif "<rp " in tag:
                assert repairs[ID][0] == True,\
                    "mid repair word start before reparandum end " + ID
                assert repairs[ID][1] == True,\
                    "mid repair word start before interregnum end " + ID
                assert repairs[ID][2] == False,\
                    "mid repair tag before repair start tag " + ID
            elif "<rpn" in tag:
                # make sure the rps is order in tag string is before
                assert repairs[ID][0] == True,\
                    "repair end before reparandum end " + ID
                assert repairs[ID][1] == True,\
                    "repair end before interregnum end " + ID
                assert repairs[ID][2] == False,\
                    "repair end before repair start " + ID
                repairs[ID][2] = True
            # do the replacement of the tag's ID after checking
            new_tags.append(tag.replace(ID, id_map[ID]))
        if normalize_ID:
            tags[i] = "".join(new_tags)
    assert all([repairs[ID][2] for ID in repairs.keys()]),\
        "Unresolved repairs:\n\t" + str(repairs)


def shuffle(lol, seed):
    """Shuffle inplace each list in the same order.

    lol :: list of list as input
    seed :: seed the shuffling
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def minibatch(l, bs):
    """Returns a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]

    l :: list of word idxs
    """
    out = [l[:i] for i in xrange(1, min(bs, len(l) + 1))]
    out += [l[i - bs:i] for i in xrange(bs, len(l) + 1)]
    assert len(l) == len(out)
    return out


def indices_from_length(sentence_length, bs, start_index=0):
    """Return a list of indexes pairs (start/stop) for each word
    max difference between start and stop equal to bs
    border cases are treated as follow:
    eg: sentenceLength=4 and bs = 3
    will output:
    [[0,0],[0,1],[0,2],[1,3]]
    """
    l = map(lambda x: start_index + x, xrange(sentence_length))
    out = []
    for i in xrange(0, min(bs, len(l))):
        out.append([l[0], l[i]])
    for i in xrange(bs + 1, len(l) + 1):
        out.append([l[i - bs], l[i - 1]])
    assert len(l) == sentence_length
    return out


def context_win(l, win):
    """Return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    given a list of indexes composing a sentence.

    win :: int corresponding to the size of the window
    """
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win / 2 * [-1] + l + win / 2 * [-1]
    out = [lpadded[i:i + win] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def context_win_backwards(l, win):
    '''Same as contextwin except only backwards context
    (i.e. like an n-gram model)
    '''
    assert win >= 1
    l = list(l)
    lpadded = (win - 1) * [-1] + l
    out = [lpadded[i: i + win] for i in range(len(l))]
    assert len(out) == len(l)
    return out


def corpus_to_indexed_matrix(my_array_list, win, bs, sentence=False):
    """Returns a matrix of contextwins for a list of utterances of
    dimensions win * n_words_in_corpus
    (i.e. total length of all arrays in my_array_list)
    and corresponding matrix of indexes (of just start/stop for each one)
    so 2 * n_words_in_corpus
    of where to access these, using bs (backprop distance)
    as the limiting history size
    """
    sentences = []  # a list (of arrays, or lists?), returned as matrix
    indices = []  # a list of index pairs (arrays?), returned as matrix
    totalSize = 0
    if sentence:
        for sent in my_array_list:
            # padding with eos
            mysent = np.asarray([-1] * (bs - 1) + list(sent))
            # get list of context windows
            mywords = context_win_backwards(mysent, win)
            # just one per utterance for now..
            cindices = [[totalSize, totalSize + len(mywords) - 1]]
            cwords = []
            for i in range(bs, len(mywords) + 1):
                words = list(itertools.chain(*mywords[(i - bs):i]))
                cwords.append(words)  # always (bs * n) words long
            # print cwords
            sentences.extend(cwords)
            indices.extend(cindices)
            totalSize += len(cwords)
    else:
        for sentence in my_array_list:
            # get list of context windows
            cwords = context_win_backwards(sentence, win)
            cindices = indices_from_length(len(cwords), bs, totalSize)
            indices.extend(cindices)
            sentences.extend(cwords)
            totalSize += len(cwords)
    for s in sentences:
        if any([x is None for x in s]):
            print s
    return np.matrix(sentences, dtype='int32'), indices


def convert_from_eval_tags_to_inc_disfluency_tags(tags, words,
                                                  representation="disf1",
                                                  limit=8):
    """Conversion from disfluency tagged corpus with xml-style tags
    as from STIR (https://bitbucket.org/julianhough/stir)
    to the strictly left-to-right schemas as
    described by Hough and Schlangen 2015 Interspeech paper,
    which are used by RNN architectures at runtime.

    Keyword arguments:
    tags -- the STIR eval style disfluency tags
    words -- the words in the utterance
    representation -- the number corresponding to the type of tagging system
    1=standard, 2=rm-N values where N does not count intervening edit terms
    3=same as 2 but with a 'c' tag after edit terms have ended.
    limit -- the limit on the distance back from the repair start
    """
    repair_dict = defaultdict(list)
    new_tags = []
    for t in range(0, len(tags)):
        if "uttseg" in representation:
            m = re.search(r'<[ct]*/>', tags[t])
            if m:
                TTO_tag = m.group(0)
            tags[t] = tags[t].replace(TTO_tag, "")
        if "dact" in representation:
            m = re.search(r'<diact type="[^\s]*"/>', tags[t])
            if m:
                dact_tag = m.group(0)
                tags[t] = tags[t].replace(dact_tag, "")
        if "laugh" in representation:
            m = re.search(r'<speechLaugh/>|<laughter/>', tags[t])
            if m:
                laughter_tag = m.group(0)
            else:
                laughter_tag = "<nolaughter/>"
            tags[t] = tags[t].replace(laughter_tag, "")
        current_tag = ""
        if "<e/>" in tags[t] or "<i" in tags[t]:
            current_tag = "<e/>"  # TODO may make this an interregnum
        if "<rms" in tags[t]:
            rms = re.findall("<rms id\=\"[0-9]+\"\/>", tags[t], re.S)
            for r in rms:
                repairID = r[r.find("=") + 2:-3]
                repair_dict[repairID] = [t, 0]
        if "<rps" in tags[t]:
            rps = re.findall("<rps id\=\"[0-9]+\"\/>", tags[t], re.S)
            for r in rps:
                repairID = r[r.find("=") + 2:-3]
                assert repair_dict.get(repairID), str(
                    repairID) + str(tags) + str(words)
                repair_dict[repairID][1] = t
                dist = min(t - repair_dict[repairID][0], limit)
                # adjust in case the reparandum is shortened due to the limit
                repair_dict[repairID][0] = t - dist
                current_tag += "<rm-{}/>".format(dist) + "<rpMid/>"
        if "<rpn" in tags[t]:
            rpns = re.findall("<rpnrep id\=\"[0-9]+\"\/>", tags[t], re.S) +\
                re.findall("<rpnsub id\=\"[0-9]+\"\/>", tags[t], re.S)
            rpns_del = re.findall("<rpndel id\=\"[0-9]+\"\/>", tags[t], re.S)
            # slight simplifying assumption is to take the repair with
            # the longest reparandum as the end category
            repair_type = ""
            longestlength = 0
            for r in rpns:
                repairID = r[r.find("=") + 2:-3]
                l = repair_dict[repairID]
                if l[1] - l[0] > longestlength:
                    longestlength = l[1] - l[0]
                    repair_type = "Sub"
            for r in rpns_del:
                repairID = r[r.find("=") + 2:-3]
                l = repair_dict[repairID]
                if l[1] - l[0] > longestlength:
                    longestlength = l[1] - l[0]
                    repair_type = "Del"
            if repair_type == "":
                raise Exception("Repair not passed \
            correctly." + str(words) + str(tags))
            current_tag += "<rpEnd" + repair_type + "/>"
            current_tag = current_tag.replace("<rpMid/>", "")
        if current_tag == "":
            current_tag = "<f/>"
        if "uttseg" in representation:
            current_tag += TTO_tag
        if "dact" in representation:
            current_tag += dact_tag
        if "laugh" in representation:
            current_tag += laughter_tag
        new_tags.append(current_tag)
    return new_tags


def convert_from_inc_disfluency_tags_to_eval_tags(
        tags, words,
        start=0,
        representation="disf1_uttseg"):
    """Converts the incremental style output tags of the RNN to the standard
    STIR eval output tags.
    The exact inverse of convertFromEvalTagsToIncrementalDisfluencyTags.

    Keyword arguments:
    tags -- the RNN style disfluency tags
    words -- the words in the utterance
    start -- position from where to begin changing the tags from
    representation -- the number corresponding to the type of tagging system,
    1=standard, 2=rm-N values where N does not count intervening edit terms
    3=same as 2 but with a 'c' tag after edit terms have ended.
    """
    # maps from the repair ID to a list of
    # [reparandumStart,repairStart,repairOver]
    repair_dict = defaultdict(list)
    new_tags = []
    if start > 0:
        # assuming the tags up to this point are already converted
        new_tags = tags[:start]
        if "mid" not in representation:
            rps_s = re.findall("<rps id\=\"[0-9]+\"\/>", tags[start - 1])
            rpmid = re.findall("<rp id\=\"[0-9]+\"\/>", tags[start - 1])
            if rps_s:
                for r in rps_s:
                    repairID = r[r.find("=") + 2:-3]
                    resolved_repair = re.findall(
                        "<rpn[repsubdl]+ id\=\"{}\"\/>"
                        .format(repairID), tags[start - 1])
                    if not resolved_repair:
                        if not rpmid:
                            rpmid = []
                        rpmid.append(r.replace("rps ", "rp "))
            if rpmid:
                newstart = start - 1
                for rp in rpmid:
                    rps = rp.replace("rp ", "rps ")
                    repairID = rp[rp.find("=") + 2:-3]
                    # go back and find the repair
                    for b in range(newstart, -1, -1):
                        if rps in tags[b]:
                            repair_dict[repairID] = [b, b, False]
                            break
    for t in range(start, len(tags)):
        current_tag = ""
        if "uttseg" in representation:
            m = re.search(r'<[ct]*/>', tags[t])
            if m:
                TTO_tag = m.group(0)
        if "<e/>" in tags[t] or "<i/>" in tags[t]:
            current_tag = "<e/>"
        if "<rm-" in tags[t]:
            rps = re.findall("<rm-[0-9]+\/>", tags[t], re.S)
            for r in rps:  # should only be one
                current_tag += '<rps id="{}"/>'.format(t)
                # print t-dist
                if "simple" in representation:
                    # simply tagging the rps
                    pass
                else:
                    dist = int(r[r.find("-") + 1:-2])
                    repair_dict[str(t)] = [max([0, t - dist]), t, False]
                    # backwards looking search if full set
                    # print new_tags, t, dist, t-dist, max([0, t-dist])
                    # print tags[:t+1]
                    rms_start_idx = max([0, t - dist])
                    new_tags[rms_start_idx] = '<rms id="{}"/>'\
                        .format(t) + new_tags[rms_start_idx]\
                        .replace("<f/>", "")
                    reparandum = False  # interregnum if edit term
                    for b in range(t - 1, max([0, t - dist]), -1):
                        if "<e" not in new_tags[b]:
                            reparandum = True
                            new_tags[b] = '<rm id="{}"/>'.format(t) +\
                                new_tags[b].replace("<f/>", "")
                        if reparandum is False and "<e" in new_tags[b]:
                            new_tags[b] = '<i id="{}"/>'.\
                                format(t) + new_tags[b]
        # repair ends
        if "<rpEnd" in tags[t]:
            rpns = re.findall("<rpEndSub/>", tags[t], re.S)
            rpns_del = re.findall("<rpEndDel/>", tags[t], re.S)
            rpnAll = rpns + rpns_del
            if rpnAll:
                for k, v in repair_dict.items():
                    if t >= int(k) and v[2] is False:
                        repair_dict[k][2] = True
                        # classify the repair
                        if rpns_del:  # a delete
                            current_tag += '<rpndel id="{}"/>'.format(k)
                            rpns_del.pop(0)
                            continue
                        reparandum = [words[i] for i in range(0, len(new_tags))
                                      if '<rms id="{}"/>'.
                                      format(k) in new_tags[i] or
                                      '<rm id="{}"/>'.
                                      format(k) in new_tags[i]]

                        repair = [words[i] for i in range(0, len(new_tags))
                                  if '<rps id="{}"/>'.format(k)
                                  in new_tags[i] or '<rp id="{}"/>'.format(k)
                                  in new_tags[i]] + [words[t]]

                        if reparandum == repair:
                            current_tag += '<rpnrep id="{}"/>'.format(k)
                        else:
                            current_tag += '<rpnsub id="{}"/>'.format(k)
        # mid repair phases still in progress
        for k, v in repair_dict.items():
            if t > int(k) and v[2] is False:
                current_tag += '<rp id="{}"/>'.format(k)
        if current_tag == "":
            current_tag = "<f/>"
        if "uttseg" in representation:
            current_tag += TTO_tag
        new_tags.append(current_tag)
    return new_tags


def verify_dialogue_data_matrix(dialogue_data_matrix, word_dict=None,
                                pos_dict=None, tag_dict=None, n_lm=0,
                                n_acoustic=0):
    """Boolean check of whether dialogue data consistent
    with args. Checks all idxs are valid and number of features is correct.
    Standard form of each row of the matrix should be:

    utt_index, word_idx, pos_idx, word_duration,
        acoustic_feats.., lm_feats....,label
    """
    l = 3 + n_acoustic + n_lm + 1  # row length
    try:
        for i, row in enumerate(dialogue_data_matrix):
            assert len(row) == l,\
                "row {} wrong length {}, should be {}".format(i, len(row), l)
            assert word_dict[row[1]] is not None,\
                "row[1][{}] {} not in word dict".format(i, row[1])
            assert pos_dict[row[2]] is not None,\
                "row[2][{}] {} not in POS dict".format(i, row[2])
            assert tag_dict[row[-1]] is not None,\
                "row[-1][{}] {} not in tag dict".format(i, row[-1])
    except AssertionError as a:
        print a
        return False
    return True


def verify_dialogue_data_matrices_from_folder(matrices_folder_filepath,
                                              word_dict=None,
                                              pos_dict=None,
                                              tag_dict=None,
                                              n_lm=0,
                                              n_acoustic=0):
    """A boolean check that the dialogue matrices make sense for the
    particular configuration in args and tag2idx dicts.
    """
    for dialogue_file in os.listdir(matrices_folder_filepath):
        v = np.load(matrices_folder_filepath + "/" + dialogue_file)
        if not verify_dialogue_data_matrix(v,
                                           word_dict=word_dict,
                                           pos_dict=pos_dict,
                                           tag_dict=tag_dict,
                                           n_lm=n_lm,
                                           n_acoustic=n_acoustic):
            print "{} failed test".format(dialogue_file)
            return False
    return True


def dialogue_data_and_indices_from_matrix(d_matrix,
                                          n_extra,
                                          pre_seg=False,
                                          window_size=2,
                                          bs=9,
                                          tag_rep="disf1_uttseg",
                                          tag_to_idx_map=None,
                                          in_utterances=False):
    """Transforming from input format of row:

    utt_index, word_idx, pos_idx, word_duration,
        acoustic_feats.., lm_feats....,label

    to 5-tuple of:

    word_idx, pos_idx, extra, labels, indices

    where :word_idx: and :pos_idx: have the correct window context
    according to @window_size
    and :indices: is the start and stop points for consumption by the
    net in training for each label in :labels:. :extra: is the matrix
    of extra features.
    """
    utt_indices = d_matrix[:, 0]
    words = d_matrix[:, 1]
    pos = d_matrix[:, 2]
    extra = None if n_extra == 0 else d_matrix[:, 3: -1]
    labels = d_matrix[:, -1]
    word_idx = []
    pos_idx = []
    current = []
    indices = []
    previous_idx = -1
    for i, a_tuple in enumerate(zip(utt_indices, words, pos, labels)):
        utt_idx, w, p, l = a_tuple
        current.append((w, p, l))
        if pre_seg:
            if previous_idx != utt_idx or i == len(labels) - 1:
                if in_utterances:
                    start = 0 if indices == [] else indices[-1][1] + 1
                    indices.append([start, start + (len(current) - 1)])
                else:
                    indices.extend(indices_from_length(len(current), bs,
                                                       start_index=len(indices)))
                word_idx.extend(context_win_backwards([x[0] for x in current],
                                                      window_size))
                pos_idx.extend(context_win_backwards([x[1] for x in current],
                                                     window_size))
                current = []
        elif i == len(labels) - 1:
            # indices = indices_from_length(len(current), bs)
            # currently a simple window of same size
            indices = [[j, j + bs] for j in range(0, len(current))]
            padding = [[-1, -1]] * (bs - window_size)
            word_idx = padding + context_win_backwards([x[0] for x in current],
                                                       window_size)
            pos_idx = padding + context_win_backwards([x[1] for x in current],
                                                      window_size)
        previous_idx = utt_idx
    return np.asarray(word_idx, dtype=np.int32), np.asarray(pos_idx,
                                                            dtype=np.int32),\
        extra,\
        labels,\
        np.asarray(indices, dtype=np.int32)


if __name__ == '__main__':
    tags = '<f/>,<rms id="3"/>,<i id="3"/><e/>,<rps id="3"/>' +\
        '<rpnsub id="3"/>,<f/>,<e/>,<f/>,' + \
        '<f/>'
    tags = tags.split(",")
    words = "i,like,uh,love,to,uh,love,alot".split(",")
    print tags
    print len(tags), len(words)
    new_tags = convert_from_eval_tags_to_inc_disfluency_tags(
        tags,
        words,
        representation="disf1")
    print new_tags
    old_tags = convert_from_inc_disfluency_tags_to_eval_tags(
        new_tags,
        words,
        representation="disf1")
    assert old_tags == tags, "\n " + str(old_tags) + "\n" + str(tags)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print context_win_backwards(x, 2)
    print "indices", indices_from_length(11, 9)
