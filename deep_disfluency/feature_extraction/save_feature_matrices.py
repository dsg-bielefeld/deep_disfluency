import os
import pandas
import numpy as np
import argparse

from deep_disfluency.load.load import load_word_rep, load_tags
from .feature_utils import load_data_from_disfluency_corpus_file
from .feature_utils import load_data_from_corpus_file
from .feature_utils import sort_into_dialogue_speakers


def open_with_pandas_read_csv(filename, header=None, delim="\t"):
    df = pandas.read_csv(filename, sep=delim, header=header, comment='f')
    # specific to these files
    data = df.values
    return data


def myround(x, base=.01, prec=2):
    return round(base * round(float(x)/base), prec)


def get_interval_indices(timings, context_frames):
    """For a given list of start and end times for intervals of
    interest (which will be words),
    return a matrix of start_time/stop times rounded up to the
    nearest window_size ms.
    """
    # print data
    # print data.shape
    # first, round up the time to the nearest int in terms of frames
    final_data = []
    for _, stop in timings:
        stop = int(100.0 * stop)  # turn into 100fps
        assert stop - (stop - context_frames) == 50,\
            stop - (stop - context_frames)
        final_data.append((stop - context_frames, stop))
    return final_data


def get_audio_features(filename, features=[0, 2, 3, 4, 5, 6, 7, 8],
                       interval_indices=None):
    """For a given file, extract the features at the indices
    specified for given
    the given intervals- return a list of tuples
    (end_time, array_of_features_for_that_interval)
    the list will be the same length as the number of intervals

    Input format:

    frameIndex; frameTime; pcm_RMSenergy_sma; pcm_LOGenergy_sma;
    F0final_sma; voicingFinalUnclipped_sma; F0raw_sma; pcm_intensity_sma;
    pcm_loudness_sma"""
    print(filename)
    data = open_with_pandas_read_csv(filename, header=True, delim=";")
    final_data = []
    data = data[:, features]  # just get the features of interest
    for interval in interval_indices:
        start, stop = interval
        # print start, stop
        # print data[start-1:stop-1]
        my_data = data[start:stop]  # to account for the header
        # print my_data
        # print my_data.shape
        if start < 0 and my_data.shape[0] < 50:
            print("beneath 0 starting context, add padding")
            padding = np.zeros((0-start, data.shape[1]))
            my_data = np.concatenate([padding, my_data])
        if my_data.shape[0] < 50:
            print("adding end padding")
            padding = np.zeros((50-my_data.shape[0], data.shape[1]))
            my_data = np.concatenate([my_data, padding])
        assert my_data.shape[0] == 50, my_data.shape[0]
        final_data.append(my_data)
    return final_data


def save_feature_matrices(target_dir,
                          dialogues,
                          use_timing_data=False,
                          word_2_idx_dict=None,
                          pos_2_idx_dict=None,
                          label_2_idx_dict=None,
                          audiofeatures_dir=None,
                          lmfeatures_dir=None,
                          ):
    """Saves features as individual dialogue matrices.
    Standard form of each row of each matrix should be:

    utt_number,
    word_idx, pos_idx, word_duration,lm_feats....,acoustic_feats...,label

    :param target_dir - location of the folder where the
    pickled matrices will be saved
    :param dialogues - list of dialogue speakers words, POS (and optionally
    timings)
    :param word_2_idx_dict - map from word to 1-hot index
    :param pos_2_idx_dict - map from POS tag to 1-hot index
    :param label_2_idx_dict - map from label to 1-hot index
    :param audiofeatures_dir - location of audio feature csv files
    :param lmfeatures_dir - location of lang model feature csv files
    """
    if audiofeatures_dir:
        assert use_timing_data
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # We're going to make the featire matrices for the experiments here
    # One per dialogue speaker
    # the line of every file/numpy pickle is
    # end_of_word_time, f1window(500ms), f2window....
    # 1=fnwindow(500ms), wordindex, posindex, labelindex
    missed = []
    unknown_words = 0
    unknown_pos = 0
    for d in dialogues:
        all_features = []
        d_name, data = d
        frames, lex_data, pos_data, indices, labels = data
        if audiofeatures_dir:
            timing_data = [x[2] for x in lex_data]
            ind = get_interval_indices(timing_data, 0.01, 50)
            audio_feature_name = audiofeatures_dir +\
                "/sw0{}.csv".format(d_name)
            audio = None
            try:
                audio = get_audio_features(audio_feature_name,
                                           interval_indices=ind)
            except IOError:
                missed.append(audio_feature_name)
                continue
        if lmfeatures_dir:
            lm_filename = lmfeatures_dir +\
                 "/sw0{}.csv".format(d_name)
            lm_feature_list = open_with_pandas_read_csv(lm_filename)
        for i in range(0, len(frames)):
            tag = labels[i]
            word_ix = word_2_idx_dict.get(lex_data[i])
            if word_ix is None:
                # print "unknown word", lex_data[i]
                word_ix = word_2_idx_dict.get("<unk>")
                unknown_words += 1
            # print word_ix
            pos_ix = pos_2_idx_dict.get(pos_data[i])
            if pos_ix is None:
                print("unknown pos", "%%%" + pos_data[i] + "%%%")
                pos_ix = pos_2_idx_dict.get("<unk>")
                unknown_pos += 1
            # print pos_ix
            # print labels[i]
            label_ix = label_2_idx_dict.get(tag)
            if label_ix is None:
                # TODO not looking at 1-shot learning for now
                print("no label for", tag)
                raise Exception
            # final_lexical = np.asarray([[word_ix, pos_ix]])\
            #    .reshape((2, 1))
            # begin filling the vector (row)
            frame_vector = [int(frames[i])]
            frame_vector.append(word_ix)
            frame_vector.append(pos_ix)
            # print label_ix
            # print audio[i].shape
            # print audio[i][:,1:].shape
            if use_timing_data:
                word_timing = timing_data[i]
                frame_vector.append(word_timing)
            if audiofeatures_dir:
                raise NotImplementedError("Add audio features.")
                # audio_d1 = audio[i][:, 1:].shape[0]
                # audio_d2 = audio[i][:, 1:].shape[1]
                # flatten:
                #final_audio = audio[i][:, 1:].reshape((audio_d1 * audio_d2, 1))
                # print final_audio.shape
                #frame_vector.append(final_audio)
            if lmfeatures_dir:
                frame_vector.extend(lm_feature_list[i])
            frame_vector.append(label_ix)
            # for fv in frame_vector:
            #    print fv
            # print "len frame vector", len(frame_vector)
            all_features.append(np.asarray(frame_vector))
        dialogue_matrix = np.concatenate([all_features])
        np.save(target_dir+"/"+d_name+".npy", dialogue_matrix)
    if missed:
        print("dialogues missed", missed)
    print(unknown_words, "unknown words in corpus")
    print(unknown_pos, "unknown pos in corpus")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine all features\
    into matrices (one per dialogue participant) and save.')
    parser.add_argument(
        '-i', action='store', dest='corpus_file',
        default='../data/disfluency_detection/\
        switchboard/swbd_disf_train_1_partial_data.csv',
        help='Location of the corpus to annotate with\
        language model features.')
    parser.add_argument(
        '-lm', action='store',
        dest='lm_feature_folder',
        default=None,
        help='The location of the LM features in csv format.')
    parser.add_argument(
        '-a', action='store',
        dest='audio_feature_folder',
        default=None,
        help='The location of the audio features in csv format.')
    parser.add_argument(
        '-m', action='store',
        dest='matrices_folder',
        default=None,
        help='Folder to save the pickled matrices in.')
    parser.add_argument(
        '-w', action='store',
        dest='word_rep_file',
        default=None,
        help='Filepath to the word representation.')
    parser.add_argument(
        '-p', action='store',
        dest='pos_rep_file',
        default=None,
        help='Filepath to the POS tag representation.')
    parser.add_argument(
        '-tag', action='store',
        dest='label_rep_file',
        default=None,
        help='Filepath to the label representation,')
    args = parser.parse_args()

    word_dict = load_word_rep(args.word_rep_file)
    pos_dict = load_word_rep(args.pos_rep_file)
    label_dict = load_tags(args.label_rep_file)

    print('tags', args.label_rep_file)
    use_timing_data = False
    if "timings" in args.corpus_file:
        dialogues = load_data_from_corpus_file(args.corpus_file)
        use_timing_data = True
    else:
        print("no timings")
        IDs, timings, seq, pos_seq, targets = \
            load_data_from_disfluency_corpus_file(
                                                args.corpus_file,
                                                convert_to_dnn_format=True)
        raw_dialogues = sort_into_dialogue_speakers(IDs,
                                                    timings,
                                                    seq,
                                                    pos_seq,
                                                    targets,
            add_uttseg="uttseg" in args.label_rep_file,
            add_dialogue_acts="dact" in args.label_rep_file
                                                    )
        dialogues = []
        for conv_no, indices, lex_data, pos_data, labels in raw_dialogues:
            frames = indices
            dialogues.append((conv_no, (frames, lex_data, pos_data, indices,
                                        labels)))

    save_feature_matrices(args.matrices_folder,
                          dialogues,
                          use_timing_data=use_timing_data,
                          word_2_idx_dict=word_dict,
                          pos_2_idx_dict=pos_dict,
                          label_2_idx_dict=label_dict,
                          audiofeatures_dir=args.audio_feature_folder,
                          lmfeatures_dir=args.lm_feature_folder)
