
import csv
import numpy
from copy import deepcopy
import argparse
from deep_disfluency.utils.tools import \
    convert_from_eval_tags_to_inc_disfluency_tags


def wer(r, h, macro=False):
    """
        Calculation of WER with Levenshtein distance.
        O(nm) time ans space complexity.

        >>> wer("who is there".split(), "is there".split())
        33.3333333
        >>> wer("who is there".split(), "".split())
        100.0
        >>> wer("".split(), "who is there".split())
        100.0

        macro :: Return the overall cost, else the WER
    """
    # initialisation
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.int)
    d = d.reshape((len(r)+1, len(h)+1))

    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    if macro:
        return d[len(r)][len(h)]
    wer = 1 if len(r) == 0 and 0 < len(h) else d[len(r)][len(h)]/float(len(r))
    return 100 * float(wer)


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


def get_turn_number(triple_string):
    triple = triple_string[1:].strip(")").split(":")
    return int(triple[0])


def interactive_dialogue(pair):
    """Return the data from both speakers in the pair as
    a single dialogue with interactive speaker tags
    for switching speakers or sticking with the speakers
    """
    speaker_1, speaker_2 = pair
    frames_data, lex_data, pos_data, indices_data, labels_data = speaker_1
    # normalize for the sort so that the turn can't change when <c is in
    # the uttseg
    for i, label in enumerate(labels_data):
        if "<t" not in label:
            indices_data[i] = indices_data[i-1]  # dynamically updates
    # put a search criteria on the first speaker
    # then sort the rest in place
    # turn into two list of tuples
    s1 = deepcopy([(f, l, p, i, y, "a") for f, l, p, i, y in zip(
        frames_data,
        lex_data,
        pos_data,
        indices_data,
        labels_data)])
    frames_data, lex_data, pos_data, indices_data, labels_data = speaker_2
    for i, label in enumerate(labels_data):
        if "<t" not in label:
            indices_data[i] = indices_data[i-1]  # dynamically updates
    s2 = deepcopy([(f, l, p, i, y, "b") for f, l, p, i, y in zip(
        frames_data,
        lex_data,
        pos_data,
        indices_data,
        labels_data)])
    all_data = sorted(s1 + s2, key=lambda x: get_turn_number(x[3]))
    previous_speaker = ""
    previous_tag = ""
    for l in range(len(all_data)):
        line = list(all_data[l])
        #print line
        speaker = 'keep'
        if not line[5] == previous_speaker:
            speaker = 'take'
            previous_speaker = line[5]
        line[4] = line[4] + '<speaker floor="{}"/>'.format(speaker)
        all_data[l] = tuple(line)
        if "t/>" in previous_tag and "<c" in line[4]:
            print("warning interactive, continuation after end")
            print(line)
        previous_tag = line[4]
    final_data = []
    for i in range(5):
        final_data.append([x[i] for x in all_data])
    return final_data


def concat_all_data_all_speakers(dialogues, interactive_sort=False,
                                 divide_into_utts=False,
                                 convert_to_dnn_format = False,
                                representation="disf_1",
                                limit=8):
    """Concatenates all the data together as lists of lists"""
    frames = []
    words = []
    pos = []
    indices = []
    labels = []
    current_pair = []
    for d in sorted(dialogues, key=lambda x: x[0]):
        dialogueID, data = d
        if interactive_sort:
            current_pair.append(deepcopy(d))
            if len(current_pair) == 2:
                assert(current_pair[0][0][:-1] == current_pair[1][0][:-1])
                print([x[0] for x in current_pair])
                data = deepcopy(interactive_dialogue(deepcopy([x[1] for x in
                                                      current_pair])))
                current_pair = []
            else:
                continue
        frames_data, lex_data, pos_data, indices_data, labels_data = data
        lex_data = [x[0][1] for x in lex_data]
        pos_data = [x[1] for x in pos_data]
        if divide_into_utts:
            frames.append(deepcopy(frames_data))
            current_lex = []
            current_pos = []
            current_idx = []
            current_label = []
            started = False
            last_idx = ""
            for i, w_i, p_i, i_i, l_i in zip(list(range(0,len(lex_data))),
                                                   lex_data,
                                                   pos_data,
                                                   indices_data,
                                                   labels_data):
                print(w_i, p_i, i_i, l_i)
                if (started and i_i.split(":")[0] != last_idx) or \
                        i == len(lex_data) - 1:
                    if convert_to_dnn_format:
                        current_label = \
                            convert_from_eval_tags_to_inc_disfluency_tags(
                                current_label,
                                current_lex,
                                representation,
                                limit)
                    words.append(deepcopy(current_lex))
                    pos.append(deepcopy(current_pos))
                    indices.append(deepcopy(current_idx))
                    labels.append(deepcopy(current_label))
                    current_lex = []
                    current_pos = []
                    current_idx = []
                    current_label = []
                current_lex.append(w_i)
                current_pos.append(p_i)
                current_idx.append(i_i)
                last_idx = i_i.split(":")[0]
                current_label.append(l_i)
                started = True
        else:
            frames.append(deepcopy(frames_data))
            words.append(deepcopy(lex_data))
            pos.append(deepcopy(pos_data))
            indices.append(deepcopy(indices_data))
            labels.append(deepcopy(labels_data))

    print("concatenated all data")
    # print [len(x) for x in [frames, words, pos, indices, labels]]
    return frames, words, pos, indices, labels


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
        if i == len(tags)-1:
            tags[i] = tags[i] + "t/>"
        else:
            tags[i] = tags[i] + "c/>"
    return tags


def sort_into_dialogue_speakers(IDs, mappings, utts, pos_tags=None,
                                labels=None,
                                add_uttseg=True,
                                add_dialogue_acts=True,
                                convert_to_dnn_tags=False):
    # for each utterance, given its ID get its conversation number
    # and dialogue participant in the format
    # needed for MSALign files
    # return a list of tuples of (dialogueID+speaker [eg. sw4004A],
    # dialogue_act
    # ((ID1,utterance)),(ID2,utterance))
    dialogue_speakers = []
    currentA = ""
    currentB = ""
    A_utts = []
    B_utts = []
    A_mappings = []
    B_mappings = []
    A_pos = []
    B_pos = []
    A_labels = []
    B_labels = []

    for ID, map, utt, pos, tags in zip(IDs, mappings, utts, pos_tags, labels):
        # if "3756" in ID:
        # print ID, mapping, utt
        split = ID.split(":")
        dialogue = split[0]
        speaker = split[1]
        uttID = split[2]
        # mapping = [uttID] * len(utt)
        dialogue_act = split[3]
        current_speaker = "".join([dialogue, speaker])
        if "A" in current_speaker:
            if current_speaker != currentA and not currentA == "":
                dialogue_speakers.append((currentA, A_mappings, A_utts, A_pos,
                                          A_labels))
                A_utts = []
                A_mappings = []
                A_pos = []
                A_labels = []
            currentA = current_speaker
            A_utts.extend(list(utt))
            A_mappings.extend(list(map))
            A_pos.extend(list(pos))
            if convert_to_dnn_tags:
                tags = convert_from_eval_tags_to_inc_disfluency_tags(tags, utt)
            if add_uttseg:
                tags = add_word_continuation_tags(tags)
            if add_dialogue_acts:
                tags = [this_tag + '<diact type="{0}"/>'
                        .format(dialogue_act) for this_tag in tags]
            A_labels.extend(tags)
        elif "B" in current_speaker:
            if current_speaker != currentB and not currentB == "":
                dialogue_speakers.append((currentB, B_mappings, B_utts, B_pos,
                                          B_labels))
                B_utts = []
                B_mappings = []
                B_pos = []
                B_labels = []
            currentB = current_speaker
            B_utts.extend(list(utt))
            B_mappings.extend(list(map))
            B_pos.extend(list(pos))
            if convert_to_dnn_tags:
                tags = convert_from_eval_tags_to_inc_disfluency_tags(tags, utt)
            if add_uttseg:
                tags = add_word_continuation_tags(tags)
            if add_dialogue_acts:
                tags = [this_tag + '<diact type="{0}"/>'
                        .format(dialogue_act) for this_tag in tags]
            B_labels.extend(tags)

    if not (currentA, A_mappings, A_utts, A_pos, A_labels) in\
            dialogue_speakers[-2:]:
        # if current_speaker != currentA and not currentA == "":
        dialogue_speakers.append((currentA, A_mappings, A_utts, A_pos,
                                  A_labels))  # concatenate them all together
    if not (currentB, B_mappings, B_utts, B_pos, B_labels) in \
            dialogue_speakers[-2:]:
        # if current_speaker != currentB and not currentB == "":
        dialogue_speakers.append((currentB, B_mappings, B_utts, B_pos,
                                  B_labels))  # concatenate them all together
    return dialogue_speakers


def load_data_from_disfluency_corpus_file(fp, representation="disf1", limit=8,
                                          convert_to_dnn_format=False):
    """Loads from file into five lists of lists of strings of equal length:
    one for utterance iDs (IDs))
    one for word timings of the targets (start,stop)
    one for words (seq), 
    one for pos (pos_seq) 
    one for tags (targets).
     
    NB this does not convert them into one-hot arrays, just outputs lists of string tags."""
    print("loading from", fp)
    f = open(fp)
    print("loading data", f.name)
    count_seq = 0
    IDs = []
    seq = []
    pos_seq = []
    targets = []
    timings = []
    currentTimings = []
    current_fake_time = 0 # marks the current fake time for the dialogue (i.e. end of word)
    current_dialogue = ""
    
    reader=csv.reader(f,delimiter='\t')
    counter = 0
    utt_reference = ""
    currentWords = []
    currentPOS = []
    currentTags = []
    current_fake_time = 0
    
    #corpus = "" # can write to file
    for ref,timing,word,postag,disftag in reader: #mixture of POS and Words
        # print ref, timing, word, postag, disftag
        counter+=1
        if not ref == "":
            if count_seq>0: #do not reset the first time
                #convert to the inc tags
                if convert_to_dnn_format:
                    currentTags = \
                        convert_from_eval_tags_to_inc_disfluency_tags(
                            currentTags,
                            currentWords,
                            representation,
                            limit)
                #corpus+=utt_reference #write data to a file for checking
                #convert to vectors
                seq.append(tuple(currentWords))
                pos_seq.append(tuple(currentPOS))
                targets.append(tuple(currentTags))
                IDs.append(utt_reference)
                timings.append(tuple(currentTimings))
                #reset the words
                currentWords = []
                currentPOS = []
                currentTags = []
                currentTimings = []
            #set the utterance reference
            count_seq+=1
            utt_reference = ref
            if not utt_reference.split(":")[0] == current_dialogue:
                current_dialogue = utt_reference.split(":")[0]
                current_fake_time = 0 #TODO fake for now- reset the current beginning of word time
        currentWords.append(word)
        currentPOS.append(postag)
        currentTags.append(disftag)
        currentTimings.append(timing)
        current_fake_time+=1
    #flush
    if not currentWords == []:
        if convert_to_dnn_format:
            currentTags = \
                convert_from_eval_tags_to_inc_disfluency_tags(
                    currentTags,
                    currentWords,
                    representation,
                    limit)
        seq.append(tuple(currentWords))
        pos_seq.append(tuple(currentPOS))
        targets.append(tuple(currentTags))
        IDs.append(utt_reference)
        timings.append(tuple(currentTimings))

    assert len(seq) == len(targets) == len(pos_seq)
    print("loaded " + str(len(seq)) + " sequences")
    f.close()
    return (IDs,timings,seq,pos_seq,targets)


def load_data_from_corpus_file(filename, limit=8,
                               representation="disf1",
                               convert_to_dnn_format=False):
    """Loads from disfluency detection with timings file.
    """
    all_speakers = []
    lex_data = []
    pos_data = []
    frames = []
    labels = []
    indices = []

    latest_increco = []
    latest_pos = []
    latest_labels = []
    latest_indices = []

    a_file = open(filename)
    started = False
    conv_no = ""
    prev_word = -1
    prev_pos = -1
    prev_index = ""
    print("loading in timings file", filename, "...")
    for line in a_file:
        if "Speaker:" in line:
            if not started:
                started = True
            else:
                # flush
                # print line
                if not latest_increco == []:
                    shift = -1
                    for i in range(0, len(latest_increco)):
                        triple = latest_increco[i]
                        if triple[1] == triple[2]:
                            # print "same timing!", triple
                            shift = i
                            break
                    if shift > -1:
                        latest_increco = fill_in_time_approximations(
                                                        latest_increco,
                                                        shift)
                    lex_data.extend(deepcopy(latest_increco))
                    pos_data.extend(deepcopy(latest_pos))
                    # convert to the disfluency tags for this
                    # latest_labels = convertFromEvalTagsToIncDisfluencyTags()
                    if convert_to_dnn_format:
                        latest_labels = \
                            convert_from_eval_tags_to_inc_disfluency_tags(
                                latest_labels,
                                latest_increco,
                                representation,
                                limit)
                    labels.extend(deepcopy(latest_labels))
                    indices.extend(deepcopy(latest_indices))
                # fake
                # print lex_data
                frames = [x[-1] for x in lex_data]  # last word end time
                # acoustic_data = [0] * len(lex_data)  # fakes
                #indices = [0] * len(lex_data)

                all_speakers.append((conv_no, (frames, lex_data, pos_data,
                                               indices, labels)))
                # reset
                lex_data = []
                pos_data = []
                labels = []
                indices = []
                latest_increco = []
                latest_pos = []
                latest_labels = []
                latest_indices = []
                prev_word = -1
                prev_pos = -1
                prev_index = ""

            conv_no = line.strip("\n").replace("Speaker: ", "")
            continue
        if line.strip("\n") == "":
            continue
        spl = line.strip("\n").split("\t")
        # print "@@@@" + line + "@@@@"
        # raw_input()
        idx = spl[0]
        start = float(spl[1])
        end = float(spl[2])
        word = spl[3]
        pos = spl[4]
        # need to convert to the right rep here
        tag = spl[5]

        latest_increco.append(([prev_word, word], start, end))
        latest_pos.append(deepcopy([prev_pos, pos]))
        latest_labels.append(tag)
        latest_indices.append(idx)
        prev_word = word
        prev_pos = pos

    # flush
    if not latest_increco == []:
        shift = -1
        for i in range(0, len(latest_increco)):
            triple = latest_increco[i]
            if triple[1] == triple[2]:
                shift = i
                break
        if shift > -1:
            latest_increco = fill_in_time_approximations(latest_increco, shift)
        lex_data.extend(latest_increco)
        pos_data.extend(latest_pos)
        if convert_to_dnn_format:
            latest_labels = \
                convert_from_eval_tags_to_inc_disfluency_tags(
                    latest_labels,
                    latest_increco,
                    representation,
                    limit)
        labels.extend(latest_labels)
        indices.extend(latest_indices)
    frames = [x[-1] for x in lex_data]  # last word end time
    # acoustic_data = [0,] * len(lex_data)  # fakes..
    # indices = [0] * len(lex_data)
    all_speakers.append((conv_no, (frames, lex_data, pos_data,
                                   indices, labels)))
    print(len(all_speakers), "speakers with timings input")
    # print "first few lengths"
    # limit = 10
    # for s in all_speakers:
    #     id, data = s
    #     print [len(x) for x in data]
    #     limit-=1
    #     if limit<0: break
    return all_speakers


def get_diff_and_new_prefix(current,newprefix,verbose=False):
    """Only get the different right frontier according to the timings
    and change the current hypotheses"""
    if verbose: 
        print("current", current)
        print("newprefix", newprefix)
    rollback = 0
    original_length = len(current)
    original_current = deepcopy(current)
    for i in range(len(current)-1,-2,-1):
        if verbose: 
            print("oooo", newprefix[0])
            if not current == []:
                print(current[i])
        if i==-1 or (float(newprefix[0][1]) >= float(current[i][2])):
            if i==len(current)-1:
                current = current + newprefix
                break
            k = 0
            marker = i+1
            for j in range(i+1,len(current)):
                if k == len(newprefix):
                    break
                if verbose: 
                    print("...", j, k, current[j], newprefix[k], len(newprefix))
                if not current[j]==newprefix[k]:
                    break
                else:
                    if verbose: print("repeat")
                    k+=1
                    marker = j+1
            rollback = original_length - marker   
            current = current[:marker] + newprefix[k:]
            newprefix = newprefix[k:]
            break
    if newprefix == []:
        rollback = 0 #just no rollback if no prefix
        current = original_current #reset the current
    if verbose: 
        print("current after call", current)
        print("newprefix after call", newprefix)
        print("rollback after call", rollback)
    return (current, newprefix, rollback)


def simulate_increco_data(frame,acoustic_data,lexical_data,pos_data):
    """For transcripts + timings, create tuples of single hypotheses
    to simulate perfect ASR at the end of each word."""
    new_lexical_data = []
    new_pos_data = []
    new_acoustic_data = []
    current_time = 0
    for my_frame, acoust, word, pos in zip(frame, acoustic_data,
                                           lexical_data, pos_data):
        #print type(my_frame), type(acoust), type(word), type(pos)
        new_lexical_data.append([(word,current_time/100,my_frame/100)])
        current_time = my_frame
        new_pos_data.append([pos])
        new_acoustic_data.append([acoust])
    return new_acoustic_data, new_lexical_data, new_pos_data 


def fill_in_time_approximations(word_timing_tuples, idx):
    """Called when the start and end time of a word are the same
    (i.e. it has no length, due to technical error), it searches forward
    in the list until
    it meets a word with a different end time, then spreads the time of
    the affected timing tuples stand and end time accordingly.

    word_timing_tuples :: list, of (word, start_time, end_time) triples
    idx :: int, the index of the first offending timing in the list
    """
    affected_indices = []
    start_time = word_timing_tuples[idx][1]
    end_time = word_timing_tuples[idx][2]
    assert start_time == end_time,\
        "Function not needed- different start and end times!"
    assert len(word_timing_tuples) > 1
    if end_time == word_timing_tuples[-1][2]:
        # this end time is the same as the latest one
        # need backward search for time
        # print "backward search"
        idx = len(word_timing_tuples)-1
        for i in range(len(word_timing_tuples)-1, -1, -1):
            affected_indices.append(i)
            idx = i
            if not word_timing_tuples[i][1] == start_time:
                start_time = word_timing_tuples[i][1]
                break
    else:
        for i in range(idx, len(word_timing_tuples)):
            affected_indices.append(i)
            if not word_timing_tuples[i][2] == start_time:
                end_time = word_timing_tuples[i][2]
                break
    total_time = end_time - start_time
    assert total_time > 0.0, str(word_timing_tuples[affected_indices[0]:]) +\
        str(idx)
    mean_len = total_time/len(affected_indices)
    for i in range(idx, idx + len(affected_indices)):
        end_time = start_time + mean_len
        word_timing_tuples[i] = (word_timing_tuples[i][0], start_time,
                                 end_time)
        start_time = end_time
    return word_timing_tuples


def process_arguments(config=None,
                      exp_id=None,
                      heldout_file="../data/disfluency_detection/switchboard/swbd_heldout_partial_data.csv",
                      test_file="../data/disfluency_detection/switchboard/swbd_test_partial_data.csv",
                      use_saved=None,
                      hmm=False,
                      verbose=True):
    """Loads arguments for an experiment from a config file
    
    Keyword arguments:
    config -- the config file location, default None
    exp_id -- the experiment ID name, default None
    """
    parser = argparse.ArgumentParser(description='This script trains a RNN for disfluency detection and saves the best models and results to disk.')
    parser.add_argument('-c', '--config', type=str,
                        help='The location of the config file.', 
                        default=config)
    parser.add_argument('-e', '--exp_id',type=str,
                        help='The experiment number from which to load arguments from the config file.', 
                        default=exp_id)
    parser.add_argument('-v', '--heldout_file',type=str,
                        help='The path to the validation file.', 
                        default=heldout_file)
    parser.add_argument('-t', '--test_file',type=str,
                        help='The path to the test file.', 
                        default=test_file)
    parser.add_argument('-m', '--use_saved_model',type=int,
                        help='Epoch number of the pre-trained model to load.', 
                        default=use_saved)
    parser.add_argument('-hmm', '--hmm',type=bool,
                        help='Whether to use hmm disfluency decoder.', 
                        default=hmm)
    parser.add_argument('-verb', '--verbose',type=bool,
                        help='Whether to output training progress.', 
                        default=verbose)
    args = parser.parse_args()
    
    header = [
        'exp_id', #experiment id
        'model', #can be elman/lstm/mt_elman/mt_lstm
        'lr', #learning rate         
        'decay', # decay on the learning rate if improvement stops
        'seed', #random seed
        'window', # number of words in the context window (backwards only for disfluency)
        'bs', # number of backprop through time steps
        'emb_dimension', # dimension of word embedding
        'nhidden', # number of hidden units
        'nepochs', # maximum number of epochs
        'train_data', #which training data
        'loss_function', #default will be nll, unlikely to change
        'reg', #regularization type
        'pos', #whether pos tags or not
        'acoustic', #whether using aoustic features or not
        'embeddings', #embedding files, if any
        'update_embeddings', #whether the embeddings should be updated at runtime or not
        'batch_size', #batch size, 'word' or 'utterance'
        'tags', #the output tag representations used
        'end_utterance' #whether we do combined end of utterance detection too
        ]
    print(header)
    if args.config:
        for line in open(args.config):
            features = line.strip("\n").split("\t")
            #print features
            if features[0] != args.exp_id: continue
            for i in range(1,len(header)):
                feat_value = features[i].strip() # if string
                #print feat_value, "feat"
                if header[i] in ['lr']: feat_value = float(feat_value)
                elif header[i] in ['decay','pos','acoustic','update_embeddings','end_utterance']: 
                    if feat_value == 'True': feat_value = True
                    else: feat_value = False
                elif header[i] in ['seed','window','bs','emb_dimension','nhidden','nepochs']:
                    feat_value = int(feat_value)
                elif feat_value == 'None':
                    feat_value = None
                setattr(args, header[i], feat_value)
    #print args
    return args

if __name__ == "__main__":
    print(wer("who is there".split(), "is there".split(), macro=True))
    print(wer("who is there".split(), "".split(), macro=True))
    print(wer("".split(), "who is there".split(), macro=True))
