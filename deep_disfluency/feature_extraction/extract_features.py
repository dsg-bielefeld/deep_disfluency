# Script which calls different scripts for differing feature extraction.
# See the top-level README.md to see where the raw data should be placed.
#
# Required (language model features derivable):
#     Transcript raw data : disfluency detection corpus (already generated by
#     deep_disfluency.corpus.DisfluencyCorpusCreator.py on swda data
# Optional (if using word timing):
#     Transcript raw data: word timings from Mississippi Swbd files (download)
# Optional (if using audio features and/or ASR):
#     Audio raw data : .sph (or .wav) files (purchase from LDC)
#     OpenSmile (software) + .conf file
# Optional (if using ASR results):
#     IBM Watson ASR account
import argparse
import sys
import subprocess
import os


def extract_features(args):
    corpusName = args.divisionFile[args.divisionFile.rfind("/") + 1:].\
        replace("_ranges.text", "")
    if args.partial_words:
        corpusName += "_partial"

    corpus_filename = corpusName + "_data"
    if args.wordAlignmentFolder:
        corpus_filename += "_timings"
    corpus_filename += ".csv"
    do = False
    if args.wordAlignmentFolder and do:
        # link the word alignments to the disfluency detection corpora
        # also adds laughter
        c = [
            sys.executable,
            os.path.dirname(os.path.realpath(__file__)) +
            '/swbd_map_word_alignments_to_SWDA_words.py',
            '-i', args.corpusLocation + "/" + corpus_filename.
                                                    replace("_timings", ""),
            '-d', args.divisionFile,
            '-a', args.wordAlignmentFolder
            ]
        if args.laughter:
            c.append('-l')
        subprocess.call(c)

    if args.newTags:
        # create the tag representations (normally from the training data
        # not allowed to look into unseen tags in the test/dev set
        c = [
            sys.executable,
            os.path.dirname(os.path.realpath(__file__)) +
            '/create_tag_files.py',
            '-i', args.corpusLocation + "/" + corpus_filename,
            '-tag', args.tagFolder,
            ]
        if args.laughter:
            c.append('-l')
        if args.uttSeg:
            c.append('-u')
        if args.dialogueActs:
            c.append('-d')
        if args.joint:
            c.append('-joint')
        subprocess.call(c)

    if args.ASR:
        raise NotImplementedError("ASR results getting script TODO.")

    if args.posTagger:
        if args.train_pos:
            raise NotImplementedError("POS tag training script TODO")
        raise NotImplementedError("POS tagger extraction needs doing")

    if args.languageModelFolder:
        c = [
            sys.executable,
            os.path.dirname(os.path.realpath(__file__)) +
            '/add_language_model_features.py',
            '-i', args.corpusLocation + "/" + corpus_filename,
            '-lm', args.languageModelFolder,
            '-f', args.matrixFolder + "/lm_matrices",
            '-order', str(3),
            '-xlm',
            # '-tag', args.tagFolder,
            ]
        if args.partial_words:
            c.append("-p")
        c.append('-e')
        if args.uttSeg:
            c.append('-u')
        print(c)
        subprocess.call(c)

    if args.audioFolder:
        raise NotImplementedError("ASR scripts TODO.")

    # with the extracted separate raw and
    # (optionally) derived (LM, audio) features
    # add these together into single vectors for DNN training format
    # TODO for now this should only run for training and heldout data.
    # test features change based on ASR results.
    c = [
        sys.executable,
        os.path.dirname(os.path.realpath(__file__)) +
        '/save_feature_matrices.py',
        '-i', args.corpusLocation + "/" + corpus_filename,
        '-m', args.matrixFolder,
        '-w', args.tagFolder + "/swbd_word_rep.csv",
        '-p', args.tagFolder + "/swbd_pos_rep.csv",
        '-tag', args.tagFolder + "/swbd_disf1_tags.csv"
        ]
    if args.languageModelFolder:
        c.append('-lm')
        c.append(args.matrixFolder + "/lm_matrices")
    if args.audioFolder:
        c.append('-a')
        c.append(args.matrixFolder + + "/audio_matrices")
    subprocess.call(c)

    extraction_results = {"POS_accuracy": None, "asr_WER": None}
    return extraction_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extraction for\
    disfluency and other tagging tasks from disfluency detection corpora and\
    raw data.')
    parser.add_argument('-i', action='store', dest='corpusLocation',
                        default='../data/disfluency_detection',
                        help='location of the disfluency\
                        detection corpus folder')
    parser.add_argument(
        '-m', action='store',
        dest='matrixFolder',
        default='../data/disfluency_detection/feature_matrices',
        help='location of the disfluency annotation csv files'
                        )
    parser.add_argument('-f', action='store', dest='divisionFile',
                        default='../data/disfluency_detection/\
                        swda_divisions_disfluency_detection/\
                        SWDisfTrain_ranges.text',
                        help='location of the file listing the \
                        files used in the corpus')
    parser.add_argument('-p', action='store_true', dest='partial_words',
                        default=False,
                        help='Whether to use partial words or not.')
    parser.add_argument('-a', action='store', dest='wordAlignmentFolder',
                        default=None,
                        help='location of the word alignment files')
    parser.add_argument('-tag', action='store', dest='tagFolder',
                        default=None,
                        help='location of the folder with the tag to\
                        tag index mapping')
    parser.add_argument('-new_tag', action='store_true', dest='newTags',
                        default=False,
                        help='Whether to save a new tag set generated from\
                        the data set to the tag folder.')
    parser.add_argument('-pos', action='store', dest='posTagger',
                        default=None, help='A POSTagger to tag the data.\
                        If None, Gold POS tags assumed.')
    parser.add_argument('-train_pos', action='store_true', dest='trainPOS',
                        default=False,
                        help='Whether to train POS a POS tagger on the data\
                        and save it at pos.')
    parser.add_argument('-u', action='store_true', dest='uttSeg',
                        default=False,
                        help='Whether to annotate with utterance segmentation\
                        tags.')
    parser.add_argument('-d', action='store_true', dest='dialogueActs',
                        default=False,
                        help='Whether to annotate with dialogue acts.')
    parser.add_argument('-l', action='store_true', dest='laughter',
                        default=False,
                        help='Whether to annotate with laughter.')
    parser.add_argument('-joint', action='store_true', dest='joint',
                        default=False,
                        help='Whether to create a joint tag set with the \
                        cross product of the tags (which appear in the data.')
    parser.add_argument('-lm', action='store', dest='languageModelFolder',
                        default=None,
                        help='Location of where to write a clean language\
                        model files out of this corpus.')
    parser.add_argument('-xlm', action='store_true',
                        dest='crossValLanguageModelTraining',
                        default=False,
                        help='Whether to use a cross language model\
                        training to be used for getting lm features on\
                        the same data.')
    parser.add_argument('-asr', action='store_true', dest='ASR',
                        default=False,
                        help='Whether to use IBM ASR to create ASR results.')
    parser.add_argument('-credentials', action='store', dest='credentials',
                        default="1841487c-30f4-4450-90bd-38d1271df295:\
                        EcqA8yIP7HBZ",
                        help="IBM Watson credentials of format username:pword")
    parser.add_argument('-audio', action='store', dest='audioFolder',
                        default=None,
                        help='location of the audio data \
                         files with .sph or wav files.')
    parser.add_argument('-opensmile', action='store', dest='openSmileConfig',
                        default=None,
                        help='location of the OpenSmile config file.')
    args = parser.parse_args()
    r = extract_features(args)
    print(r)
