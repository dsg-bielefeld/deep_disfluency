import re


def uttseg_pattern(tag):
    trp_tag = ""
    if "<c" in tag:  # i.e. a continuation of utterance from the previous word
        trp_tag += "c_{}"
    elif "<t" in tag:  # i.e. a first word of an utterance
        trp_tag += "t_{}"
    if "c/>" in tag:  # i.e. predicts a continuation next
        trp_tag += "_c"
    elif "t/>" in tag:  # i.e. predicts an end of utterance for this word
        trp_tag += "_t"
    assert trp_tag != "" and trp_tag[0] != "_" and trp_tag[-1] != "}",\
        "One or both TRP tags not given " + str(trp_tag) + " for:" + tag
    return trp_tag


def convert_to_disfluency_tag(previous, tag):
    """Returns (dis)fluency tag which is dealt with in a uniform way by
    the Markov model.
    """
    if not previous:
        previous = ""
    if "<f" in tag:
        if "rpSM" in previous or "rpM" in previous:
            # TODO change back for later ones without MID!
            return "rpM"  # can be mid repair
        return "f"
    elif "<e" in tag and "<i" not in tag:
        if "rpM" in previous or "rpSM" in previous or 'eR' in previous:
            #  Not to punish mid-repair
            # return "eR"  # edit term mid-repair phase, not interreg
            return "e"
        return "e"
    elif "<i" in tag:
        return "i"
    elif "<rm-" in tag:
        if "<rpEnd" in tag:
            return "rpSE"
        return "rpSM"
    elif "<rpMid" in tag:
        return "rpM"
    elif "<rpEnd" in tag:
        return "rpE"
    print "NO TAG for" + tag


def convert_to_uttseg_tag(previous, tag):
    """Returns plain uttseg tags."""
    return uttseg_pattern(tag).format("w")


def convert_to_disfluency_uttseg_tag(previous, tag):
    """Returns joint a list of (dis)fluency and trp tag which is dealt with
    in a uniform way by the Markov model.
    """
    if not previous:
        previous = ""
    trp_tag = uttseg_pattern(tag)
    if "<f" in tag:
        if "rpSM" in previous or "rpM" in previous:
            # TODO change back for later ones without MID!
            if "<t" not in tag and "t/>" not in tag:
                return trp_tag.format("rpM")  # can be mid repair
        return trp_tag.format("f")
    elif "<e" in tag and "<i" not in tag:
        if "rpM" in previous or "rpSM" in previous or 'eR' in previous:
            # Not to punish mid-repair
            if "t/>" not in tag:
                # edit term mid-repair phase, not interreg
                # return trp_tag.format("eR")
                return trp_tag.format("e")
        return trp_tag.format("e")
    elif "<i" in tag:
        return trp_tag.format("i")  # This should always be c_i_c
    elif "<rm-" in tag:
        if "<rpEnd" in tag:
            return trp_tag.format("rpSE")
        return trp_tag.format("rpSM")
    elif "<rpMid" in tag:
        return trp_tag.format("rpM")
    elif "<rpEnd" in tag:
        return trp_tag.format("rpE")
    print "NO TAG for" + tag


def convert_to_disfluency_tag_simple(previous, tag):
    if not previous:
        previous = ""
    if "<f" in tag:
        return "f"
    elif "<i" in tag:
        return "i"  # This should always be c_i_c when in uttseg
    elif "<e" in tag:
        return "e"
    elif "<rm-" in tag:
        return "rpSE"
    print "NO TAG for" + tag


def convert_to_disfluency_uttseg_tag_simple(previous, tag):
    """Returns joint a list of (dis)fluency and trp tag which is dealt with in
    a uniform way by the Markov model.
    Simpler version with fewer classes.
    """
    if not previous:
        previous = ""
    trp_tag = uttseg_pattern(tag)
    return trp_tag.format(convert_to_disfluency_tag_simple(previous, tag))


def convert_to_diact_tag(previous, tag):
    """Returns the dialogue act.
    """
    if not previous:
        previous = ""
    diact = ""
    m = re.search(r'<diact type="([^\s]*)"/>', tag)
    if m:
        diact = m.group(1)
        return diact
    print "NO TAG for" + tag


def convert_to_diact_uttseg_tag(previous, tag):
    """Returns joint a list of dialgoue and trp tag which is dealt with in
    a uniform way by the Markov model.
    """
    # print previous, tag
    if not previous:
        previous = ""
    trp_tag = uttseg_pattern(tag)
    return trp_tag.format(convert_to_diact_tag(previous, tag))


def convert_to_diact_interactive_tag(previous, tag):
    """Returns the dialogue act but with the fact it is keeping or
    taking the turn.
    """
    if not previous:
        previous = ""
    diact = convert_to_diact_tag(previous, tag)
    m = re.search(r'speaker floor="([^\s]*)"/>', tag)
    if m:
        return diact + "_" + m.group(1)
    print "NO TAG for" + tag


def convert_to_diact_uttseg_interactive_tag(previous, tag):
    """Returns the dialogue act but with the fact it is keeping or
    taking the turn.
    """
    if not previous:
        previous = ""
    trp_tag = uttseg_pattern(tag)
    return trp_tag.format(convert_to_diact_interactive_tag(previous, tag))


def convert_to_source_model_tags(seq, uttseg=True):
    """Convert seq of repair tags to the source model tags.
    Source model tags are:
    <s/> start fluent word (utteseg only)
    <f/> fluent mid utterance word
    <e/> edited word
    """
    source_tags = []
    # print seq
    for i, tag in enumerate(seq):
        if "<e" in tag or "<i" in tag:
            source_tags.append("<e/>")
        elif "<rm-" in tag:
            m = re.search("<rm-([0-9]+)\/>", tag)
            if m:
                back = min([int(m.group(1)), len(source_tags)])
                # print back, i, source_tags
                for b in range(i - back, i):
                    source_tags[b] = "<e/>"
                source_tags.append("<f/>")
            else:
                raise Exception('NO REPARANDUM DEPTH {}'.format(tag))
        else:
            if "<t" in tag:
                source_tags.append("<s/>")
                continue
            source_tags.append("<f/>")
    return source_tags
