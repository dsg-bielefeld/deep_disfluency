# -*- coding: utf-8 -*-
"""Methods to convert BNC POS tags to Switchboard ones, and merge
"do n't" > "dont" and combine their POS tags.
Creating a clean corpus of line and POS
"""
import logging
tag_separator = '_@'


def get_word(tag_and_word):
    x = tag_and_word.split(tag_separator)
    try:
        return x[1]
    except IndexError:
        logging.warning(
            '{0} is not tagged. \
            This is not a problem now because \
             we are just returning the word.\n'.format(tag_and_word))
        return x[0]


def get_tag(tag_and_word):
    x = tag_and_word.split(tag_separator)
    return x[0]


def translate_to_PennPOS(pos, pos2, word, word2):
    """Translate BNC POS tags to SWDA/Penn standard POS tags."""
    # need big mapping function from BNC to Penn
    if pos in ('AJ0', 'ORD'):
        pos = 'JJ'
    elif pos in ('AJC'):
        pos = 'JJR'
    elif pos in ('AJS'):
        pos = 'JJS'
    elif pos in ('AT0'):
        pos = 'DT'
    elif pos in ('AV0', 'RB'):
        pos = 'RB'
    elif pos in ('AVP', 'RP'):
        pos = 'RP'
    elif pos in ('AVQ'):
        pos = 'WRB'
    elif pos in ('CJC'):
        pos = 'CC'
    elif pos in ('CJS', 'CJT'):
        pos = 'IN'
    elif pos in ('CRD'):
        pos = 'CD'
    elif pos in ('DT0'):
        pos = 'DT'
    elif pos in ('DTQ'):
        pos = 'WDT'
    elif pos in ('DPS'):
        pos = 'PRP$'
    elif pos in ('EX0'):
        pos = 'EX'
    elif pos in ('ITJ'):
        pos = "UH"
    elif pos in ('NN0', 'NN1'):
        pos = 'NN'
    elif pos in ('NNS', "NN2"):
        pos = 'NNS'
    elif pos in ('NP0'):
        pos = 'NNP'  # NO NNPS equiv in the BNC
    elif pos in ('PNI'):
        pos = 'NN'
    elif pos in ('PRP', 'PNP', 'PNX', 'PRF'):
        pos = 'PRP'
    elif pos in ('PNQ'):
        pos = 'WP'
    elif pos in ('POS'):
        pos = "POS"
    elif pos.startswith("PU"):
        pos = pos
    elif pos in ('TO0') and (not word2 is None and pos2.startswith("V")):
        pos = "IN"
    elif pos in ('TO0'):
        pos = "TO"
    elif pos in ('UNC'):
        pos = "SYM"  # JUST MAKING UNKNOWN ONES SYMBOLS
    elif (pos in ('VBB') and word in ("'re", 'am', 'are')):
        # not quite right. non-third person sing. pres need to convert
        pos = "VBP"
    elif pos in ("VBB", "VDB", "VHB", "VVB"):
        # collapsing these into 3rd person sing. present, could be VB too?
        pos = "VB"
    elif pos in ('VBI', 'VDI', "VHI", "VVI"):
        pos = "VB"  # INFINITIVE
    elif pos in ('VBD', 'VDD', "VHD", "VVD"):
        pos = "VBD"  # PAST TENSE
    elif pos in ("VBG", 'VDG', "VHG", "VVG"):
        pos = "VBG"  # GERUND
    elif pos in ("VBN", 'VDN', "VHN", "VVN"):
        pos = "VBN"  # past participle
    elif pos in ("VBZ") and word in ("'s"):
        pos = "BES"  # 's,
    elif pos in ("VHZ") and word in ("'s"):
        pos = "HVS"  # john's got flu.
    elif pos in ("VBZ", "VDZ", "VHZ", "VVZ"):
        pos = "VBZ"  # is,'s, does,'s for does
    elif pos in ("VM0"):
        pos = "MD"
    elif pos in ("XX0"):
        pos = "RB"  # do n't/not
    elif pos in ("ZZ0"):
        pos = "SYM"
    elif "-" in pos:
        pos = translate_to_PennPOS(pos[0:pos.rfind("-")], pos2, word, word2)
    else:
        print "UNIDENTIFIED " + pos + word

    return pos


def convert_BNC_sentence(sentence, posconvert=True):
    """Takes BNC input, makes POS tags to be equivalent to Switchboard
    also compresses words together which were separated
    morphologically "don't", "isn't" etc.
    """
    tokensorig = sentence.split()
    tokens = []
    postokens = []
    # list of the tokens we want to mark as Unknown?
    dodgy = ["NNP", "NNPS", "CD", "LS", "SYM", "FW"]
    continuations = ["s", "re", "ve", "m", "nt"]
    for i in range(0, len(tokensorig)):
        # print get_word(token)  #take out /u's... and sentence markers, will
        # put these back in..
        token = tokensorig[i]
        nexttoken = None
        nextpos = None
        if i < len(tokensorig) - 1:
            nexttoken = tokensorig[i + 1].replace(".", "").\
                replace("?", "").replace(";", "").replace(
                "!", "").replace("(", "").replace(")", "").\
                replace('"', "").replace("/", "").lower()
            nextpos = get_tag(tokensorig[i + 1])

        word = get_word(token).lower()

        # do unknown word here, or remove?
        pos = get_tag(token)
        if pos.startswith("PU"):
            continue
        continuation = False
        word = word.replace(',', '').replace(".", "").\
            replace("?", "").replace(";", "").replace("!", "").\
            replace("(", "").replace(")", "").replace('"', "").replace("/", "")
        if pos == "POS" or word in continuations:
            continuation = True
        if posconvert:
            pos = translate_to_PennPOS(pos, nextpos, word, nexttoken)
        word = word.replace("'", "").replace(
            "%", "percent").lower()  # get rid of 's
        if continuation == True:
            if len(tokens) > 1:
                tokens[-1] = tokens[-1] + word
                postokens[-1] = postokens[-1] + pos
                continue
        # TODO check consistency, not neccessarily, this is their unknown
        # words, not ours
        elif pos in dodgy or "\u" in word:
            word = word.replace("\u", "")
            # not differentiating numbers for now, could get this from CD
            # (cardinal numbers), above
            word = "$unc$" + word

        if not word == "." and not word == "," and not word == "" \
                and not pos.startswith("PU"):
            tokens.append(word)
            postokens.append(pos)

    # print "rektokenizing:"
    # print tokens #means we don't have to check else where
    assert len(tokens) == len(postokens)

    POS = ""
    words = ""
    for i in range(0, len(tokens)):
        POS += postokens[i] + " "
        words += tokens[i] + " "
    POS = POS[:-1] + "\n"
    words = words[:-1] + "\n"
    return words, POS


if __name__ == '__main__':
    bnc_file = open(
        "../data/raw_data/bnc_spoken/BNC_spoken_tags-and-words.CORPUS", "r")
    converted_file_REF = open("../data/bnc_spoken/bnc_spokenREF.text", "w")

    for line in bnc_file:
        # print line
        words, POS = convert_BNC_sentence(line, posconvert=False)
        converted_file_REF.write("," + words)
        converted_file_REF.write("POS," + POS)

    bnc_file.close()
    converted_file_REF.close()
