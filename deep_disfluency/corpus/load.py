# -*- coding: utf-8 -*-

# Methods to load in both language model and disfluency detection corpora,
# and to split them
import csv


def get_tag_data_from_corpus_file(f):
    """Loads from file into four lists of lists of strings of equal length:
    one for utterance iDs (IDs))
    one for words (seq), 
    one for pos (pos_seq) 
    one for tags (targets)."""

    f = open(f)
    print("loading data", f.name)
    count_seq = 0
    IDs = []
    seq = []
    pos_seq = []
    targets = []

    reader = csv.reader(f, delimiter='\t')
    counter = 0
    utt_reference = ""
    currentWords = []
    currentPOS = []
    currentTags = []

    # corpus = "" # can write to file
    for ref, _, word, postag, disftag in reader:  # mixture of POS and Words
        counter += 1
        if not ref == "":
            if count_seq > 0:  # do not reset the first time
                # convert to the inc tags
                # corpus+=utt_reference #write data to a file for checking
                # convert to vectors
                seq.append(tuple(currentWords))
                pos_seq.append(tuple(currentPOS))
                targets.append(tuple(currentTags))
                IDs.append(utt_reference)
                # reset the words
                currentWords = []
                currentPOS = []
                currentTags = []
            # set the utterance reference
            count_seq += 1
            utt_reference = ref
        currentWords.append(word)
        currentPOS.append(postag)
        currentTags.append(disftag)
    # flush
    if not currentWords == []:
        seq.append(tuple(currentWords))
        pos_seq.append(tuple(currentPOS))
        targets.append(tuple(currentTags))
        IDs.append(utt_reference)

    assert len(seq) == len(targets) == len(pos_seq)
    print("loaded " + str(len(seq)) + " sequences")
    f.close()
    return (IDs, seq, pos_seq, targets)


def lm_corpus_splits(corpus_name, split=0.0, pos_tagged=True):
    """Build the word lm and pos tag lm (if specified) training corpora from 
    the STIR lm style corpora.
    Returns a 4-tuple of train, pos_train, heldout, pos_heldout
    Some of which can be None, if no POS and/or split==0.0 (no heldout)
    """
    # convert to a split
    split = (100.0 - split) / 100.0

    train, pos_train, heldout, pos_heldout = None, None, None, None
    # print "splitting", corpus_name
    corpus = open(corpus_name)
    reader = csv.reader(corpus, delimiter=',')
    lines = []
    pos_lines = []
    id_s = []
    unique_id_s = []
    train = ""
    heldout = ""
    for mode, text in reader:
        if mode == "REF":  # don't bother with the indices
            continue
        if mode == "POS":
            if pos_tagged:
                pos_lines.append(text)
            continue
        # get here we're on the word level
        lines.append(text)
        dialogue_no = mode.split(":")[0]
        id_s.append(dialogue_no)
        if not dialogue_no in unique_id_s:
            unique_id_s.append(dialogue_no)
    last_train = id_s.index(unique_id_s[int(split * len(unique_id_s)) - 1])
    # make the split in terms of number of files
    train = "\n".join(lines[0:last_train + 1])
    if split < 1.0:
        heldout = "\n".join(lines[last_train + 1:])
    if pos_tagged:
        pos_train = "\n".join(pos_lines[0:last_train + 1])
        if split < 1.0:
            pos_heldout = "\n".join(pos_lines[last_train + 1:])
    corpus.close()
    # print "splitting done"
    return train, pos_train, heldout, pos_heldout
