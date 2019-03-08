import csv
from .corpus_util import add_word_continuation_tags


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
    mappings = []

    reader = csv.reader(f, delimiter='\t')
    counter = 0
    utt_reference = ""
    currentWords = []
    currentPOS = []
    currentTags = []
    currentMappings = []

    # corpus = "" # can write to file
    for ref, map, word, postag, disftag in reader:  # mixture of POS and Words
        counter += 1
        if not ref == "":
            if count_seq > 0:  # do not reset the first time
                # convert to the inc tags
                # corpus+=utt_reference #write data to a file for checking
                # convert to vectors
                seq.append(tuple(currentWords))
                pos_seq.append(tuple(currentPOS))
                targets.append(tuple(add_word_continuation_tags(currentTags)))
                IDs.append(utt_reference)
                mappings.append(tuple(currentMappings))
                # reset the words
                currentWords = []
                currentPOS = []
                currentTags = []
                currentMappings = []
            # set the utterance reference
            count_seq += 1
            utt_reference = ref
        currentWords.append(word)
        currentPOS.append(postag)
        currentTags.append(disftag)
        currentMappings.append(map)
    # flush
    if not currentWords == []:
        seq.append(tuple(currentWords))
        pos_seq.append(tuple(currentPOS))
        targets.append(tuple(add_word_continuation_tags(currentTags)))
        IDs.append(utt_reference)
        mappings.append(tuple(currentMappings))

    assert len(seq) == len(targets) == len(pos_seq)
    print("loaded " + str(len(seq)) + " sequences")
    f.close()
    return (IDs, mappings, seq, pos_seq, targets)


def sort_into_dialogue_speakers(IDs, mappings, utts, pos_tags=None,
                                labels=None):
    """For each utterance, given its ID get its conversation number and
    dialogue participant in the format needed for word alignment files.

    Returns a list of tuples:
    (speaker, mappings, utts, pos, labels)

    """
    dialogue_speaker_dict = dict()  # keys are speaker IDs of filename:speaker
    # vals are tuples of (mappings, utts, pos_tags, labels)
    current_speaker = ""

    for ID, mapping, utt, pos, label in zip(IDs,
                                            mappings,
                                            utts,
                                            pos_tags,
                                            labels):
        split = ID.split(":")
        dialogue = split[0]
        speaker = split[1]
        # uttID = split[2]
        current_speaker = "-".join([dialogue, speaker])
        if current_speaker not in list(dialogue_speaker_dict.keys()):
            dialogue_speaker_dict[current_speaker] = [[], [], [], []]

        dialogue_speaker_dict[current_speaker][0].extend(list(mapping))
        dialogue_speaker_dict[current_speaker][1].extend(list(utt))
        dialogue_speaker_dict[current_speaker][2].extend(list(pos))
        dialogue_speaker_dict[current_speaker][3].extend(list(label))
    # turn into 5-tuples
    dialogue_speakers = [(key,
                          dialogue_speaker_dict[key][0],
                          dialogue_speaker_dict[key][1],
                          dialogue_speaker_dict[key][2],
                          dialogue_speaker_dict[key][3])
                         for key in sorted(dialogue_speaker_dict.keys())]
    return dialogue_speakers


def write_corpus_file_add_fake_timings_and_utt_tags(f, target_path,
                                                    verbose=False):
    target_file = open(target_path, "w")
    IDs, mappings, utts, pos_tags, labels = get_tag_data_from_corpus_file(f)
    dialogue_speakers = sort_into_dialogue_speakers(IDs,
                                                    mappings,
                                                    utts,
                                                    pos_tags,
                                                    labels)
    for speaker_name, mapping, utt, pos, label in dialogue_speakers:
        if verbose:
            print("*" * 30)
            print(speaker_name)
            print(mapping)
            print(utt)
            print(pos)
            print(label)
            y = input()
            if y == "y":
                quit()
        target_file.write("Speaker: " + speaker_name + "\n")
        starts = list(range(0, len(label)))
        ends = list(range(1, len(label)+1))
        for m, s, e, w, p, l in zip(mapping, starts, ends, utt, pos, label):
            l = "\t".join([m, str(float(s)), str(float(e)), w, p, l])
            target_file.write(l + "\n")
        target_file.write("\n")
    target_file.close()


if __name__ == "__main__":
    #f = "../../../stir/python/data/bnc_spoken/BNC-CH_partial_data.csv"
    #write_corpus_file_add_fake_timings_and_utt_tags(
    #    f, f.replace("_data", "_data_timings"))
    f = "../../../stir/python/data/pcc/PCC_test_partial_data.csv"
    write_corpus_file_add_fake_timings_and_utt_tags(
        f, f.replace("_data", "_data_timings"))
    if False:
        f = "../../../stir/python/data/pcc/PCC_test_partial_data_old.csv"
        target = open(f.replace("_data_old", "_data"), "w")
        f = open(f)
        reader = csv.reader(f, delimiter='\t')
        for ref, map, word, postag, disftag in reader:  # mixture of POS and Words
            if not ref == "":
                spl = ref.split(":")
                n_ref = ":".join([spl[0], spl[2], spl[1]])
                target.write("\t".join([n_ref, map, word, postag, disftag]) + "\n")
            else:
                target.write("\t".join([ref, map, word, postag, disftag]) + "\n")
        target.close()
        