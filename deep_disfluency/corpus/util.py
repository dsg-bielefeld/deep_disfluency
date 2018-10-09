# -*- coding: utf-8 -*-
import re
from collections import defaultdict


def add_word_continuation_tags(tags):
    """Returns list with continuation tags for each word:
    <cc/> continues current dialogue act and the next word will also continue
    <ct/> continues current dialogue act and is the last word of it
    <tc/> starts this dialogue act tag and the next word continues it
    <tt/> starts and ends dialogue act (single word dialogue act)
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


def get_tags(s, open_delim='<',
             close_delim='/>'):
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


def strip_disf_tags_from_easy_read(text):
    """List of strings (words or POS tags) without the disfluency markup
    """
    words = []
    for w in text.split(" "):
        words.append(w[w.rfind(">") + 1:])
    return words


def disf_tags_from_easy_read(text):
    """List of disfluency tags from the inline easy read marked up utterances
    """
    tags = []
    for w in text.split():
        tags.append(w[:w.rfind(">") + 1])
    return [tag.replace("_", " ") for tag in tags]


def easy_read_disf_format(words, tags):
    """Easy read style inline disfluency tagged string."""
    final_tags = []
    for i in range(0, len(words)):
        final_tags.append("".join([tags[i].replace(" ", "_"), words[i]]))
    return " ".join(final_tags)


def detection_corpus_format(uttRef, words, pos, tags, indices):
    """Replace blanks with fluent <f/> tags and outputs tag separated."""
    for i in range(0, len(tags)):
        if tags[i] == "":
            tags[i] = "<f/>"
    final_string = "\t".join(
        [uttRef, indices.pop(0), words.pop(0), pos.pop(0), tags.pop(0)]) + "\n"
    print len(indices), len(words), len(pos), len(tags)
    print indices
    print words
    print pos
    print tags
    for i in range(0, len(tags)):
        final_string += "\t".join(["", indices[i],
                                   words[i], pos[i], tags[i]]) + "\n"
    return final_string.rstrip("\n")


def detection_corpus_format_from_easy_read(easyReadString):
    """Converts the easy read format to the detection corpus format"""
    lines = [x.split(",") for x in easyReadString.split("\n")]
    uttRef = lines[0][0]
    wordstring = lines[0][1]
    posstring = lines[1][1]
    indexstring = lines[2][1]
    tags = disf_tags_from_easy_read(wordstring)
    words = strip_disf_tags_from_easy_read(wordstring)
    pos = strip_disf_tags_from_easy_read(posstring)
    indices = indexstring.split(" ")
    return detection_corpus_format(uttRef, words, pos, tags, indices)


def easy_read_format_from_detection_corpus(detectionString):
    """The inverse function to detectionCorpusFormatStringFromEasyReadFormat.
    Mainly for checking consistency at corpus creation time.
    """
    lines = detectionString.rstrip("\n").split("\n")
    uttref = lines[0].split("\t")[0]
    lines[0] = lines[0].replace(uttref, "")
    indices = [line.split("\t")[1] for line in lines]
    words = [line.split("\t")[2] for line in lines]
    pos = [line.split("\t")[3] for line in lines]
    tags = [line.split("\t")[4].replace("<f/>", "") for line in lines]
    final_string = uttref + "," + easy_read_disf_format(words, tags) + '\n'
    final_string += "POS," + easy_read_disf_format(pos, tags) + "\n"
    final_string += "REF," + " ".join(indices)
    return final_string


def get_edit_terms_from_easy_read(text, postext):
    """Outputs tuples of each string of consecutive edit terms and their POS"""
    words = strip_disf_tags_from_easy_read(text)
    pos = strip_disf_tags_from_easy_read(postext)
    tags = disf_tags_from_easy_read(text)
    current_edit_term = ""
    current_pos_edit_term = ""
    # a list of tuples of (edit term strings, POS tags of that string)
    edit_terms = []
    for t in range(0, len(tags)):
        tag = tags[t]
        if "<e" in tag or "<i" in tag:
            current_edit_term += words[t] + " "
            current_pos_edit_term += pos[t] + " "
        elif not current_edit_term == "":  # we've built up a string, save it
            edit_terms.append(
                (current_edit_term.strip(), current_pos_edit_term.strip()))
            current_edit_term = ""
            current_pos_edit_term = ""
    if not current_edit_term == "":  # flush
        edit_terms.append(
            (current_edit_term.strip(), current_pos_edit_term.strip()))
    return edit_terms


def verify_disfluency_tags(tags, normalize_ID=False):
    """Check that the repair tags sequence is valid.

    Keyword arguments:
    normalize_ID -- boolean, whether to convert the
    repair ID numbers to be derivable from
    their unique RPS position in the utterance.
    """
    id_map = dict()  # map between old ID and new ID
    # in first pass get old and new IDs
    for i in range(0, len(tags)):
        rps = re.findall("<rps id\=\"[0-9]+\"\/>", tags[i])
        if rps:
            id_map[rps[0][rps[0].find("=") + 2:-3]] = str(i)
    # key: old repair ID, value, list [reparandum,interregnum,repair] all True
    # when repair is all there
    repairs = defaultdict(list)
    for r in id_map.keys():
        repairs[r] = [None, None, None]  # three valued None<False<True
    # print repairs
    # second pass verify the validity of the tags and (optionally) modify the
    # IDs
    for i in range(0, len(tags)):  # iterate over all tag strings
        new_tags = []
        if tags[i] == "":
            all([repairs[ID][2] or repairs[ID] == [None, None, None]
                 for ID in repairs.keys(
            )]), "Unresolved repairs at fluent tag\n\t" + str(repairs)
        # iterate over all tags in this tag string
        for tag in get_tags(tags[i]):
            # print i, tag
            if tag == "<e/>":
                new_tags.append(tag)
                continue
            ID = tag[tag.find("=") + 2:-3]
            if "<rms" in tag:
                assert repairs[ID][0] == None, \
                    "reparandum started parsed more than once " + ID
                assert repairs[ID][1] == None, \
                    "reparandum start again during interregnum phase " + ID
                assert repairs[ID][2] == None, \
                    "reparandum start again during repair phase " + ID
                repairs[ID][0] = False  # set in progress
            elif "<rm " in tag:
                assert repairs[ID][0] != None, \
                    "mid reparandum tag before reparandum start " + ID
                assert repairs[ID][2] == None, \
                    "mid reparandum tag in a interregnum phase or beyond " + ID
                assert repairs[ID][2] == None, \
                    "mid reparandum tag in a repair phase or beyond " + ID
            elif "<i" in tag:
                assert repairs[ID][0] != None, \
                    "interregnum start before reparandum start " + ID
                assert repairs[ID][2] == None, \
                    "interregnum in a repair phase " + ID
                if repairs[ID][1] == None:  # interregnum not reached yet
                    repairs[ID][0] = True  # reparandum completed
                repairs[ID][1] = False  # interregnum in progress
            elif "<rps" in tag:
                assert repairs[ID][0] != None, \
                    "repair start before reparandum start " + ID
                assert repairs[ID][1] != True, \
                    "interregnum over before repair start " + ID
                assert repairs[ID][2] == None, \
                    "repair start parsed twice " + ID
                repairs[ID][0] = True  # reparanudm complete
                repairs[ID][1] = True  # interregnum complete
                repairs[ID][2] = False  # repair in progress
            elif "<rp " in tag:
                assert repairs[ID][0] == True, \
                    "mid repair word start before reparandum end " + ID
                assert repairs[ID][1] == True, \
                    "mid repair word start before interregnum end " + ID
                assert repairs[ID][2] == False, \
                    "mid repair tag before repair start tag " + ID
            elif "<rpn" in tag:
                # make sure the rps is order in tag string is before
                assert repairs[ID][0] == True, \
                    "repair end before reparandum end " + ID
                assert repairs[ID][1] == True, \
                    "repair end before interregnum end " + ID
                assert repairs[ID][2] == False, \
                    "repair end before repair start " + ID
                repairs[ID][2] = True
            # do the replacement of the tag's ID after checking
            new_tags.append(tag.replace(ID, id_map[ID]))
        if normalize_ID:
            tags[i] = "".join(new_tags)
    assert all([repairs[ID][2] for ID in repairs.keys()]
               ), "Unresolved repairs:\n\t" + str(repairs)


def orthography_normalization(word, pos, spelling_dict, lang='en'):
    """Converts the spelling from the transcripts into
    one that is consistent for disfluency detection.
    Filled pauses are treated specially to make
    sure the POS tags are correct.
    """
    if lang == 'en':
        um = "um"
        uh = "uh"
    elif lang == 'de':
        um = "ähm"
        uh = "äh"
    else:
        raise NotImplementedError(
            'No filled pause normalization for lang: ' + lang)

    for key in spelling_dict.keys():
        if re.match(key, word):
            word = spelling_dict[key]
            # make sure filled pauses have the right POS tags
            if word in [uh, um]:
                pos = 'UH'
            break
    return word, pos


def clean(myString):
    myString = re.sub(r"([\+/\}\[\]]\#|\{\w)", "", myString)
    elicitcharacters = "\#)(+\/[]_><,.\"\*%!=}{"
    mynewString = ""
    for char in myString:
        if not char in elicitcharacters:
            mynewString += char
    if mynewString == "":
        return None
    else:
        return mynewString.lower()


def parse_list(string):
    # returns list [1,2] from "[1,2]"
    chars = string
    Number1 = ""
    Number2 = ""
    x = False
    y = True
    for char in chars:
        if char == " ":
            continue
        if char == "[":
            x = True
        elif char == ",":
            x = False
            y = True
        elif char == "]":
            y = False
        elif x == True:
            Number1 += char
        elif y == True:
            Number2 += char
    return [int(Number1), int(Number2)]


def remove_repairs(tags, repairIDs):
    """Return a list of tags without the repairs with IDs in repairIDs."""
    for t in range(0, len(tags)):
        new_tag = tags[t]
        for repair_class in ["rms", "rm", "i", "rps", "rp",
                             "rpnsub", "rpndel", "rpnrep"]:
            for repairID in repairIDs:
                if repair_class == "i":
                    interreg = re.findall(
                        '<{} id="{}"/>'.format(repair_class, repairID),
                        new_tag)
                new_tag = new_tag.replace(
                    '<{} id="{}"/>'.format(repair_class, repairID), "")
                if (repair_class == "i" and len(interreg) > 0) \
                        and not ("<e" in new_tag or "<i" in new_tag):
                    # assure edit terms are maintained
                    new_tag += "<e/>"
        tags[t] = new_tag
    return tags


def find_repair_end(repair, disfluencyTagList):
    """Given a repair object and a disfluency tag list,
    find the repair word and tag it in place in the list.
    """
    # print "searching for repair in same utt", repair.repairID
    # loop stops at first element that's not an <i>
    for B in range(len(disfluencyTagList) - 1, -1, -1):
        if str(repair.repairID) in disfluencyTagList[B]:  # gets deletes/subs
            repair_class = repair.classify()
            disfluencyTagList[B] = disfluencyTagList[B]\
                .replace('<rp id="{}"/>'.format(repair.repairID), "")
            if repair_class == "del" and not \
                    '<rps id="{}"/>'.format(repair.repairID) \
                    in disfluencyTagList[B]:
                disfluencyTagList[B] = disfluencyTagList[B] + \
                    '<rps id="{}"/>'.format(repair.repairID)
            disfluencyTagList[B] = disfluencyTagList[B] + \
                '<rpn{} id="{}"/>'.format(repair_class, repair.repairID)
            repair.complete = True
            # print "completing" + str(repair.repairID)
            return True
    return False


def find_repair_ends_and_reclassify(problem_rpns, tag_list, word_list,
                                    search_start, partial_disallowed=False):
    """Backwards search to find a possible repair end rpn tag and
    re-classify its type if needs be.
    Return the repair ends successfully found.

    problem_rpns :: list of repair end tags (non-deletes) which are to
    be moved back before an edit term.
    tag_list :: the disfluency tags for utterance
    word_list :: the words for the utterance
    search_start :: position in utterance where backwards search starts
    non_partial :: repair end cannot be a partial word
    """
    resolved = []
    unresolved = []
    for i in range(search_start, -1, -1):
        if "<e/>" in tag_list[i]:
            continue  # only allow non-edit term words
        if partial_disallowed and word_list[i][-1] == "-":
            continue  # in partial_disallowed setting, no partial word rpns
        # if we have got here we may have a possible repair end word
        for rpn in problem_rpns:
            if rpn in resolved or rpn in unresolved:
                continue
            rpMid = rpn.replace("rpnsub", "rp").replace("rpnrep", "rp")
            rpStart = rpn.replace("rpnsub", "rps").replace("rpnrep", "rps")
            # a legit rp tag, can be the repair end
            if rpMid in tag_list[i] or rpStart in tag_list[i]:
                # get rid of rp mid tags
                tag_list[i] = tag_list[i].replace(rpMid, "")
                tag_list[i] = tag_list[i] + rpn  # add repair end tag
                # reclassify it as either repeat or substitution by iterating
                # up to this current word
                rmMid = rpn.replace("rpnsub", "rm").replace("rpnrep", "rm")
                rmStart = rpn.replace("rpnsub", "rms").replace("rpnrep", "rms")
                reparandum = []
                repair = []
                for check in range(0, i + 1):
                    if rmStart in tag_list[check] or rmMid in tag_list[check]:
                        reparandum.append(word_list[check])
                    if rpStart in tag_list[check] or rpMid in tag_list[check]:
                        repair.append(word_list[check])
                    if rpn in tag_list[check]:
                        repair.append(word_list[check])
                        # it was marked as a repeat, change if no longer a
                        # repeat
                        if "rep" in rpn:
                            if not reparandum == repair:
                                tag_list[check] = tag_list[check].replace(
                                    rpn, rpn.replace("rpnrep", "rpnsub"))
                        # else if marked as a sub, change if it is now a repeat
                        elif reparandum == repair:
                            tag_list[i] = tag_list[i].replace(
                                rpn, rpn.replace("rpnsub", "rpnrep"))
                        break
                resolved.append(rpn)  # this is a resolved repair end
    return resolved


def find_repair_end_in_previous_utts(repair, overallTagList, uttlist):
    # print "searching back for repair", repair.repairID
    testTagList = None
    testUttCaller = None
    search = 0
    # backwards search if not found- needs to be called at the end too as a
    # flush
    while not repair.complete == True:
        search += 1
        if search >= len(overallTagList):
            print 'Repair not found!'
            raise Exception
        # print "search " + str(search)
        # get the tag list *search* utterances back
        testTagList = overallTagList[-search]
        testUttCaller = uttlist[-search][2]
        # continue backtracking if not the caller
        if not testUttCaller == repair.caller:
            continue
        # search back in the previous utterance
        repair.complete = find_repair_end(repair, testTagList)
        # list mutable, so will change this here in place
        overallTagList[-search] = testTagList
    return


def find_delete_interregna_and_repair_onsets(tag_list, problem_rpns_del,
                                             interreg_start):
    """problem_rpns_del ::  list of delete repairs (consisting of their
    identifying <rpndel id="x"/>.
    tag_list :: list of disfluency tags where reparanda of those repairs
    is marked.
    interreg_start :: int, where the interregnum is known to start
    for these tags

    For each repair in problems_rpns_del mark the interregnum and
    repair onset/repair end delete word for that repair if possible.
    Return a list with those repairs with successfully resolved
    interregna and repair stats.
    """
    interreg_index_dict = defaultdict(
        list)  # key repair onset tag, value list of indices for interregnum
    resolved = []
    for i in range(interreg_start, len(tag_list)):
        tag = tag_list[i]
        for r in problem_rpns_del:
            if r in resolved:
                continue
            # interrengum could still be in there for the tag
            if r.replace("rpndel", "i") in tag:
                # remove as repair start may not be found
                tag_list[i] = tag_list[i].replace(r.replace("rpndel", "i"), "")
                interreg_index_dict[r].append(i)
            elif "<e" in tag:  # not marked as an interregnum for this repair
                interreg_index_dict[r].append(i)
            else:
                tag_list[i] += r.replace("rpndel", "rps") + r
                # print "interregs found for r",r,interreg_index_dict[r]
                # if rps found, mark its interregna
                for interreg in interreg_index_dict[r]:
                    tag_list[interreg] = r.replace(
                        "rpndel", "i") + tag_list[interreg]
                    # tag_list[interreg] =
                    # tag_list[interreg].replace("<e/>","")#turning into
                    # interregnum
                resolved.append(r)
    return resolved


def find_interregna_and_repair_onsets(tag_list, problem_rps, interreg_start,
                                      word_list):
    """For each repair in problems_rps mark the interregnum and
    repair onset/repair end delete word for that repair if possible.
    Return a list with those repairs with successfully
    resolved interregna and repair stats.

    problem_rps ::  list of repair ends (consisting of
    their identifying <rps id="x"/>.
    tag_list :: list of disfluency tags where reparanda of those
    repairs is marked.
    interreg_start :: int, where the interregnum is known to start
    for these tags
    """
    # key repair onset tag, value a list of indices for the interregnum
    interreg_index_dict = defaultdict(list)
    resolved = []
    for i in range(interreg_start, len(tag_list)):
        tag = tag_list[i]
        for r in problem_rps:
            if r in resolved:
                continue
            # interrengum could still be in there for the tag
            if r.replace("rps", "i") in tag:
                # remove as repair start may not be found
                tag_list[i] = tag_list[i].replace(r.replace("rps", "i"), "")
                interreg_index_dict[r].append(i)
            elif "<e" in tag:  # not marked as an interregnum for this repair
                interreg_index_dict[r].append(i)
            elif r.replace("rps", "rp") in tag or r.replace("rps", "rpnsub")\
                    in tag or r.replace("rps", "rpnrep") in tag:
                tag_list[i] = tag_list[i].replace(
                    r.replace("rps", "rp"), "")  # first remove any rps
                # if "<rps" in tag_list[i] : continue #don't add if embedded
                tag_list[i] = r + tag_list[i]  # add the repair start
                # now check if classification has changed
                reparandum = []
                repair = []
                for check in range(0, len(word_list)):
                    if r.replace("rps", "rms") in tag_list[check] or \
                            r.replace("rps", "rm") in tag_list[check]:
                        reparandum.append(word_list[check])
                    if r in tag_list[check] or r.replace("rps", "rp") \
                            in tag_list[check]:
                        repair.append(word_list[check])
                    if r.replace("rps", "rpnrep") in tag_list[check]:
                        repair.append(word_list[check])
                        if not reparandum == repair:
                            tag_list[i] = tag_list[i].replace(
                                r.replace("rps", "rpnrep"),
                                r.replace("rps", "rpnsub"))
                        break
                    elif r.replace("rps", "rpnsub") in tag_list[check]:
                        repair.append(word_list[check])
                        if reparandum == repair:
                            tag_list[i] = tag_list[i].replace(
                                r.replace("rps", "rpnsub"),
                                r.replace("rps", "rpnrep"))
                        break

                # print "interregs found for r",r,interreg_index_dict[r]
                # if rps found, mark its interregna
                for interreg in interreg_index_dict[r]:
                    tag_list[interreg] = r.replace(
                        "rps", "i") + tag_list[interreg]
                    # tag_list[interreg] =
                    # tag_list[interreg].replace("<e/>","")#turning into
                    # interregnum
                resolved.append(r)
    return resolved


def remove_non_edit_interregna(tags, words, problem_interreg_IDs):
    """Where an interregnum is marked but is not an edit term,
    convert the interregnum to <rp(s) repair tags.
    """
    phase_dict = dict()  # repair mapped to the phase it is currently in
    # list of 2 lists with reparandum and repair words
    phase_words_dict = defaultdict(list)
    for p in problem_interreg_IDs:
        phase_dict[p] = "o"  # initialize as o for original utterance
        phase_words_dict[p] = [[], []]
    for t in range(0, len(tags)):
        for repairID in problem_interreg_IDs:
            if '<rps id="{}"/>'.format(repairID) in tags[t]:
                # repair phase already reached, replace start with rps
                if phase_dict[repairID] == "rp":
                    tags[t] = tags[t].replace(
                        '<rps id="{}"/>'.format(repairID),
                        '<rp id="{}"/>'.format(repairID))
                else:
                    phase_dict[repairID] = "rp"
            if '<rms id="{}"/>'.format(repairID) in tags[t]:
                phase_dict[repairID] = "rm"
            # reparandum not reached yet or finished
            if phase_dict[repairID] == "o":
                continue
            if '<i id="{}"/>'.format(repairID) in tags[t]:
                if not "<e" in tags[t]:
                    # repair phase not reached yet, repair onset
                    if phase_dict[repairID] == "rm":
                        tags[t] = tags[t].replace(
                            '<i id="{}"/>'.format(repairID),
                            '<rps id="{}"/>'.format(repairID))
                        phase_dict[repairID] = "rp"
                    # repair phase reached, just repair word
                    elif phase_dict[repairID] == "rp":
                        if not '<rps id="{}"/>'.format(repairID) in tags[t]:
                            tags[t] = tags[t].replace(
                                '<i id="{}"/>'.format(repairID),
                                '<rp id="{}"/>'.format(repairID))
                        else:
                            # repair onset word from above, just get rid of
                            # interregnum
                            tags[t] = tags[t].replace(
                                '<i id="{}"/>'.format(repairID), "")

                else:
                    # while potentially a good interrengum,
                    # not if repair phase has already been reached,
                    # so leave it as a plain edit term
                    if phase_dict[repairID] == "rp":
                        tags[t] = tags[t].replace(
                            '<i id="{}"/>'.format(repairID), "")
            if phase_dict[repairID] == "rm" and not '<i id="{}"/>'.format(
                    repairID) in tags[t] and not "<e/>" in tags[t]:
                phase_words_dict[repairID][0].append(words[t])
            if phase_dict[repairID] == "rp" and not "<e/>" in tags[t] and \
                    not '<rpndel id="{}"/>'.format(repairID) in tags[t]:
                phase_words_dict[repairID][1].append(words[t])
            # reclassify if end of repair
            if '<rpndel id="{}"/>'.format(repairID) in tags[t]:
                # either a rep or sub, replace delete with appropriate class
                if len(phase_words_dict[repairID][1]) > 0:
                    if phase_words_dict[repairID][0] + [words[t]] == \
                            phase_words_dict[repairID][1]:
                        tags[t] = tags[t].replace(
                            '<rpndel id="{}"/>'.format(repairID),
                            '<rpnrep id="{}"/>'.format(repairID))
                    else:
                        tags[t] = tags[t].replace(
                            '<rpndel id="{}"/>'.format(repairID),
                            '<rpnsub id="{}"/>'.format(repairID))
                # either way get rid of rp from above
                tags[t] = tags[t].replace('<rp id="{}"/>'.format(repairID), "")
                phase_dict[repairID] = "o"
            elif '<rpnrep id="{}"/>'.format(repairID) in tags[t]:
                # if not the same, change to sub, else leave
                if phase_words_dict[repairID][0] != \
                        phase_words_dict[repairID][1]:
                    tags[t] = tags[t].replace('<rpnrep id="{}"/>'.format(
                        repairID), '<rpnsub id="{}"/>'.format(
                        repairID)).\
                        replace('<rp id="{}"/>'.
                                format(repairID), "")
                # either way get rid of rp from above
                tags[t] = tags[t].replace('<rp id="{}"/>'.
                                          format(repairID), "")
                phase_dict[repairID] = "o"
            elif '<rpnsub id="{}"/>'.format(repairID) in tags[t]:
                # if the same, change to rep, else leave
                if phase_words_dict[repairID][0] == \
                        phase_words_dict[repairID][1]:
                    tags[t] = tags[t].replace(
                        '<rpnsub id="{}"/>'.format(repairID),
                        '<rpnrep id="{}"/>'.format(repairID))
                # either way get rid of rp from above
                tags[t] = tags[t].replace('<rp id="{}"/>'.format(repairID), "")
                phase_dict[repairID] = "o"
    return


def remove_partial_words(tagList, wordsList, POSList, refList):
    """Consistent with the standard switchboard disfluency detection task,
    remove partial words,
    and any repairs whose reparanda consist solely of partial words.
    """
    repairsToRemoveNoReparandumStart = []
    repairsToRemoveNoRepairStart = []
    repairsToRemoveNoRepairEnd = []
    repairsToRemove = []
    wordsToRemove = []
    for w in range(0, len(wordsList)):
        word = wordsList[w]
        # print word, w
        if word[-1] == "-":
            wordsToRemove.append(w)
            if '<rms' in tagList[w]:
                # reparandum start cut off, store to see if it can be resolved
                # after this point
                problem_rms = re.findall(
                    "<rms id\=\"[0-9]+\"\/>", tagList[w], re.S)
                for r in problem_rms:
                    test = r.replace("rms", "rps")
                    repairsToRemoveNoReparandumStart.append(test)
            if '<rps' in tagList[w]:
                # repair start cut off
                problem_rps = re.findall(
                    "<rps id\=\"[0-9]+\"\/>", tagList[w], re.S)
                repairsToRemoveNoRepairStart.extend(problem_rps)
                problem_rpndels = re.findall(
                    "<rpndel id\=\"[0-9]+\"\/>", tagList[w], re.S)
                # if delete, try to find a non-partial word after this one,
                # shift rps+rpndel to this word if so
                for r in problem_rpndels:
                    test = r.replace("rpndel", "rps")
                    for n in range(w + 1, len(wordsList)):
                        if "<rps" in tagList[n]:
                            break  # don't make it an embedded repair
                        if not wordsList[n][-1] == "-" and \
                                not "<e/>" in tagList[n]:
                            tagList[n] = test + r + tagList[n]
                            repairsToRemoveNoRepairStart.remove(test)
                            break
            if "<rpn" in tagList[w]:
                # repair end is being cut off, see if it can be moved back, and
                # reclassify it if needs be
                problem_rpns = [
                    rpn for rpn in get_tags(tagList[w]) if "<rpn" in rpn]
                resolved = find_repair_ends_and_reclassify(
                    problem_rpns, tagList, wordsList, w - 1,
                    partial_disallowed=True)
                for r in problem_rpns:
                    test = r.replace("rpn", "rps")
                    if not r in resolved:
                        repairsToRemoveNoRepairEnd.append(test)
            # all problems resolved here, apart from interregnum onsets, which
            # should not cause problems
            continue
        # get here we have non-partial (complete) words,
        # see if problem repairs can be resolved
        # try to shift repairs missing reparandum start forward
        if '<rm ' in tagList[w]:
            # print "repairs to remove no rm", repairsToRemoveNoReparandumStart
            rm = re.findall("<rm id\=\"[0-9]+\"\/>", tagList[w], re.S)
            for r in rm:
                test = r.replace("rm", "rps")
                # we have a legit rm word
                if test in repairsToRemoveNoReparandumStart:
                    # print "deleting rm for rms"
                    # shift the rms tag along one
                    tagList[w] = tagList[w].replace(r, r.replace("rm", "rms"))
                    # print tagList[w]
                    repairsToRemoveNoReparandumStart.remove(test)
        # try to shift repair phases with their repair onset word missing along
        # one
        if '<rp ' in tagList[w]:
            rp = re.findall("<rp id\=\"[0-9]+\"\/>", tagList[w], re.S)
            for r in rp:
                test = r.replace("rp", "rps")
                # we have a legit rp word
                if test in repairsToRemoveNoRepairStart:
                    if "<rps" in tagList[w]:
                        # do not allow embedded repairs at all
                        repairsToRemove.append(test)
                        continue
                    # replace the rp tag with the rps one along one
                    tagList[w] = tagList[w].replace(r, test)
                    repairsToRemoveNoRepairStart.remove(test)
        # try to shift repair phases forward
        if '<rpnsub ' in tagList[w] or '<rpnrep ' in tagList[w]:
            # subsitution or repeat repair
            rpn = re.findall("<rpnrep id\=\"[0-9]+\"\/>", tagList[w], re.S) + \
                re.findall(
                "<rpnsub id\=\"[0-9]+\"\/>", tagList[w], re.S)
            for r in rpn:
                test = r.replace("rpnrep", "rps").replace("rpnsub", "rps")
                # we have a legit rp word
                if test in repairsToRemoveNoRepairStart:
                    if "<rps" in tagList[w]:
                        # do not allow embedded repairs at all
                        repairsToRemove.append(test)
                        continue
                    # make this word the rps one, keeping the end the same
                    tagList[w] = test + tagList[w]  # add the repair start
                    repairsToRemoveNoRepairStart.remove(test)

    repairsToRemove.extend(repairsToRemoveNoReparandumStart +
                           repairsToRemoveNoRepairStart +
                           repairsToRemoveNoRepairEnd)
    repairIDs = []
    for problem in repairsToRemove:
        repairID = problem[problem.find("=") + 2:-3]
        repairIDs.append(repairID)
    tagList = remove_repairs(tagList, repairIDs)
    i = len(tagList) - 1
    while i >= 0:
        if i in wordsToRemove:
            del tagList[i]
            del wordsList[i]
            del POSList[i]
            del refList[i]
        i -= 1
    return tagList, wordsList, POSList, refList


def classify_repair(reparandum, repair, continuation):
    """min edit distance string aligner for repair-> reparandum
    Creates table of all possible edits, only considers the paths
    from i,j=0 to i=m, j= n
    returns mapping from i to j with the max alignment- problem,
    there may be several paths. Weights:
        rep(string1,string2) 0
        repPartial(string1,string2)  1         j- john
        repReversePartial(string1,string2) 1    john j-
        samePOS(string1,string2) 2
        insertIntoReparandum(eps,string) 4
        deleteFromReparandum(string1,eps) 4
        samePOSclass(string1,string2) 5
        arbitarySubstitution(string1,string2) 7
    """
    # we have the normal del, insert and subs for min edit distance
    # we only need define the sub relation for the special cases
    # pointers
    left = "<"
    up = "^"
    diag = "\\"
    m = len(reparandum)
    n = len(repair)
    reparandum = [("", "")] + list(reparandum)  # add the empty strings
    repair = [("", "")] + list(repair)
    # print "initial:"
    # print reparandum
    # print repair

    # 2 tuples (word,POS) #initial = the table's content initialised as
    # [i,j,"",currentScore,""]
    def sub(source, goal, initial):
        if source[0] == goal[0]:
            return initial[0:2] + ["REP"] + [initial[3], diag]  # NO COST
        elif source[0][-1] == "-" and source[0][:-1] in goal[0]:
            return initial[0:2] + ["REP_complete"] + [initial[3] + 1, diag]
        elif goal[0][-1] == "-" and goal[0][:-1] in source[0]:
            return initial[0:2] + ["REP_partial"] + [initial[3] + 1, diag]
        elif source[1] == goal[1]:
            return initial[0:2] + ["SUB_POS"] + [initial[3] + 2, diag]
        elif source[1][0] == goal[1][0]:
            return initial[0:2] + ["SUB_POS_CLASS"] + [initial[3] + 5, diag]
        else:
            return initial[0:2] + ["S_ARB"] + [initial[3] + 7, diag]

    def delete(source, initial):
        category = "DEL"
        if source[0][-1] == "-":
            category += "_partial"
        return initial[0:2] + ["DEL"] + [initial[3] + 4, up]

    def insert(goal, initial):
        category = "INS"
        if goal[0][-1] == "-":
            category += "_partial"
        return initial[0:2] + ["INS"] + [initial[3] + 4, left]

    # initilisation of axes in table, hash from number to number to list (cell)
    D = []  # the cost table
    ptr = []  # the pointer table
    for i in range(0, m + 1):
        D.append([0] * (n + 1))
        ptr.append([[]] * (n + 1))  # these are mutable, just dummies

    # defaultdict(defaultdict(list))
    # the pointer table with a list of (pointer,relation) pairs
    # populate each of the table axes
    D[0][0] = 0
    j = 0
    for i in range(1, m + 1):
        a = delete(reparandum[i], [i, j, "", D[i - 1][j], ""])
        D[i][j] = a[3]  # delete cost
        ptr[i][j] = [(a[-1], a[2])]  # delete type
    i = 0
    for j in range(1, n + 1):
        a = insert(repair[j], [i, j, "", D[i][j - 1], ""])
        D[i][j] = a[3]  # insert cost
        ptr[i][j] = [(a[-1], a[2])]  # insert type

    # for i in range(0,m+1):
    #    print D[i]

    # for i in range(0,m+1):
    #    print ptr[i]

    # main recurrence relation algorithm
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # print "%%%%%%%"
            # print i
            # print j
            deltest = delete(reparandum[i], [i, j, "", D[i - 1][j], ""])
            # print deltest
            instest = insert(repair[j], [i, j, "", D[i][j - 1], ""])
            # print instest
            subtest = sub(
                reparandum[i], repair[j], [i, j, "", D[i - 1][j - 1], ""])
            # print subtest
            # print "%%%%%%%"
            # get the min cost set
            mincostset = set()
            mincostset.add(tuple(deltest))
            mincost = deltest[-2]
            tests = [instest, subtest]  # check the others
            for t in tests:
                if t[-2] < mincost:
                    mincost = t[-2]
                    mincostset = set()
                    mincostset.add(tuple(t))
                elif t[-2] == mincost:
                    mincostset.add(tuple(t))
            # add the pointers and their alignments
            ptr[i][j] = []
            for a in mincostset:
                #    print a
                ptr[i][j].append((a[-1], a[2]))
            D[i][j] = mincost
    # print the optimal alignment(s) backtrace-
    # there should only be one given the weights as
    # we shouldn't allow an ins+del to beat an arbsub
    # return a list of the alignemnts
    # gets them backwards then returns the reverse
    # print "cost = " + str(D[m][n])
    # for i in range(0,m+1):
    #    print D[i]

    # for p in range(0,m+1):
    #    print ptr[p]

    # return all and rank by best first approach
    # if there is a branch, follow and pop the first pointer, effectively
    # removing the path
    def backtrace(D, ptr, i, j, mymap, mymaps):
        if i == 0 and j == 0:  # should always get there directly
            mymaps.append(mymap)
            return
        arrow = ptr[i][j][0][0]  # get the first one
        alignment = ptr[i][j][0][1]
        score = D[i][j]
        if len(ptr[i][j]) > 1:  # more than one!
            del ptr[i][j][0]  # remove it before copying and recursing
            #mymapcopy = list(mymap)
            backtrace(D, ptr, i, j, list(mymap), mymaps)
            #ptr[i][j] = filter(lambda x: not x[0] == "\\", ptr[i][j])
            # coarse approximation
        mymap.insert(
            0, tuple([max(0, i - 1), max(0, j - 1), alignment, score]))

        if arrow == "\\":
            backtrace(D, ptr, i - 1, j - 1, mymap, mymaps)
        elif arrow == "^":
            backtrace(D, ptr, i - 1, j, mymap, mymaps)
        elif arrow == "<":
            backtrace(D, ptr, i, j - 1, mymap, mymaps)

    def rank(mymaps, start, n):
        tail = []
        for j in range(start, n):
            bestscores = []
            if len(mymaps) == 1:
                return mymaps + tail
            # should this recurse to the last mapping to j (i.e. highest value
            # for i)? yes
            for mymap in mymaps:
                for mapping in mymap:
                    if mapping[1] == j:
                        bestscore = mapping[3]
                    elif mapping[1] > j:
                        break
                bestscores.append(bestscore)  # should always get one!
            best = min(bestscores)
            # print "best"
            # print best
            # maintain all the best for further sorting; separately sort the
            # tail?
            i = 0
            a = 0
            while i < len(bestscores):
                #    print bestscores[i]
                if bestscores[i] > best:
                    tail.append(list(mymaps[a]))  # bad score
                    del mymaps[a]
                else:
                    a += 1
                i += 1
            if len(tail) > 0:
                tail = rank(tail, j, n)  # recursively sort the tail
        # print "warning no difference!!"
        return mymaps  # if no difference just return all

    mymaps = []
    mymap = []
    backtrace(D, ptr, m, n, mymap, mymaps)
    if len(mymaps) > 1:
        # print "ranking"
        # print len(mymaps)
        # print mymaps
        # sorts the list by best first as you pass left to right in the repair
        mymaps = rank(mymaps, 0, n)
    # for mapping in mymaps:
    #    print mapping
    # print "returning:"
    # print mymaps[0]
    return mymaps[0]  # only returns top, can change this


def graph_viz_repair(maps, reparandum, repair, continuation):
    """Returns a graph viz .dot input file for a
    digraph that can be rendered by graphviz
    """
    assert isinstance(reparandum, list)
    assert isinstance(repair, list)
    assert isinstance(continuation, list)
    digraphInit = """digraph Alignment {\n
    rankdir=LR;\n
    node[color=white]\n;
    """
    reparandumClusterInit = """subgraph cluster_reparandum {\n
    label = "reparandum";\n
    style = "invisible";\n
    node [color=white];\n
    edge[weight=5,constrained=true];\n"""

    repairClusterInit = """subgraph cluster_repair{\n
    label = "repair";\n
    style = "invisible";\n
    node [color=white];\n
    edge[weight=5,constrained=true];\n"""

    reparandumIndex = 0
    repairIndex = 0

    reparandumNodes = ""
    reparandumSequence = ""
    if len(repair) == 0:
        if len(continuation) == 0:
            raw_input("no continuation for rep in classify")
        repair = [continuation[0]]  # add the first one
    for i in range(len(reparandum)):
        reparandumSequence += str(i)
        reparandumNodes += str(i) + \
            "[label=\"" + reparandum[i][0].lower() + "\"];\n"
        if i < len(reparandum) - 1:
            reparandumSequence += " -> "
        else:
            reparandumSequence += ";"
    repairNodes = ""
    repairSequence = ""
    for i in range(len(repair)):
        repairSequence += "r" + str(i)
        repairNodes += "r" + \
            str(i) + "[label=\"" + repair[i][0].lower() + "\"];\n"
        if i < len(repair) - 1:
            repairSequence += " -> "
        else:
            repairSequence += ";"

    # if repeats or subs, they need the same rank, otherwise deletes a bit
    # tricky
    ranks = ""
    alignments = ""
    for alignment in maps:
        if not alignment[2] == "DEL" and not alignment[2] == "INSERT":
            ranks += """{rank="same";""" + \
                str(alignment[0]) + "; r" + str(alignment[1]) + "}\n"
        alignments += str(alignment[0]) + " -> " + "r" + str(alignment[1]) + \
            """[label=\"""" + \
            alignment[2].lower() + """\",color=red,dir=back];\n"""

    """ #aim is to produce something in this format:
    digraph Alignment {
    compound=true
    rankdir=LR;


    node[color=white];
    {rank="same";0; r0}
    {rank="same";1; r4}
    {rank="same";2; r5}


    subgraph cluster_reparandum {
    label = "reparandum";
    style=invisble;
    node [color=white];
    edge[weight=5,constrained=true];
    0[label="john"];
    1[label="likes"];
    2[label="mary"];
    edge[weight=15];
    0 -> 1 -> 2;
    }

    subgraph cluster_repair{
    label = "repair";
    style=invisible;
    node [color=white];
    edge[weight=5,constrained=true];
    r0[label="john"];
    r1[label="really"];
    r2[label="really"];
    r3[label="really"];
    r4[label="loves"];
    r5[label="mary"];
    edge[weight=15];
    r0 -> r1 -> r2 -> r3 -> r4 -> r5;
    }

    edge[constrained=false]
    0 -> r0[label="rep",color=red,dir=back];
    0 -> r1[label="insert",color=red,dir=back];
    0 -> r2[label="insert",color=red,dir=back];
    0 -> r3[label="insert",color=red,dir=back];
    1 -> r4[label="sublexical",color=red,dir=back];
    2 -> r5[label="rep",color=red,dir=back];

    }
    """
    finalResult = digraphInit + ranks + reparandumClusterInit + \
        reparandumNodes + "edge[weight=15];\n" + reparandumSequence + "\n}\n\n"\
        + repairClusterInit + repairNodes + \
        "edge[weight=15];\n" + repairSequence + "\n}\n\n" + \
        "edge[constrained=false]" + alignments + "\n}\n"
    return finalResult


if __name__ == '__main__':
    #s = SelfRepair()
    reparandum = [("there", "EX"), ("were", "VBD")]
    repair = [("they", "PRP")]
    continuation = [("a", "DT")]
    #repair = [("You","NNP"),("really","RB"),("like","VP"),("him","NP")]
    #reparandum = [("Y-","NNP"),("like","VP"),("john","NN")]
    # repair = [("Y-","NNP"),("like","VP"),("and","cc"),("I","RB"),
    #("like","VP"),("I","RB"),("like","VP"),("john","NN")]
    graph_viz_repair(classify_repair(
        reparandum, repair, [("", "")]), reparandum, repair, continuation)
