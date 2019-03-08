"""
Script to run the experiments described in:

Julian Hough and David Schlangen.
Recurrent Neural Networks for Incremental Disfluency Detection.
INTERSPEECH 2015.

"""
import sys
import subprocess
import os
import tarfile
import zipfile
import urllib.request, urllib.parse, urllib.error
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../../")
from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger


# The data must been downloaded
# and put in place according to the top-level README
# each of the parts of the below can be turned off
# though they must be run in order so the latter stages work
download_raw_data = False
create_disf_corpus = False
extract_features = False
train_models = False
test_models = True

asr = False  # extract and test on ASR results too
partial = False  # whether to include partial words or not

range_dir = THIS_DIR + \
    '/../data/disfluency_detection/swda_divisions_disfluency_detection'
file_divisions_transcripts = [
    ('train', range_dir + '/swbd_disf_train_1_ranges.text'),
    # range_dir + '/swbd_disf_train_audio_ranges.text',
    ('heldout', range_dir + '/swbd_disf_heldout_ranges.text'),
    ('test', range_dir + '/swbd_disf_test_ranges.text'),
]

SWBD_TIMINGS_URL = 'http://www.isip.piconepress.com/' + \
    'projects/switchboard/releases/ptree_word_alignments.tar.gz'

SWDA_CORPUS_URL = 'https://github.com/julianhough/' + \
    'swda/blob/master/swda.zip?raw=true'

SWBD_TIMINGS_DIR = THIS_DIR + '/../data/raw_data/' + \
    SWBD_TIMINGS_URL.split('/')[-1].replace(".tar.gz", "")

SWDA_CORPUS_DIR = THIS_DIR + '/../data/raw_data/' + \
    SWDA_CORPUS_URL.split('/')[-1].replace(".zip", "")

# the experiments in the Interspeech paper
# 18 non-POS window length 2
# 21 POS length 2 RNN
# 23 POS length 3 RNN
# 41 POS length 2 LSTM  # not in paper, for comparison
# experiments = [18, 21, 23, 41]
experiments = [21, 41]  # reduced version for speed for now

# 1. download the data
if download_raw_data:
    name = THIS_DIR + '/../data/raw_data/swda.zip'
    if not os.path.isfile(name):
        print('downloading', name)
        urllib.request.urlretrieve(SWDA_CORPUS_URL, name)
        zipf = zipfile.ZipFile(name)
        zipf.extractall(path=SWDA_CORPUS_DIR)
        zipf.close()
        print('extracted at', SWDA_CORPUS_DIR)

    name = THIS_DIR + '/../data/raw_data/' + SWBD_TIMINGS_URL.split('/')[-1]
    if not os.path.isfile(name):
        print('downloading', name)
        urllib.request.urlretrieve(SWBD_TIMINGS_URL, name)
        tar = tarfile.open(name)
        tar.extractall(path=SWBD_TIMINGS_DIR)
        tar.close()
        print('extracted at', SWBD_TIMINGS_DIR)


# 1. Create the base disfluency tagged corpora in a standard format
"""
for all divisions call the corpus creator
parse c line parameters
Optional arguments:
-i string, path of source data (in swda style)
-t string, target path of folder for the preprocessed data
-f string, path of file with the division of files to be turned into
a corpus
-a string, path to disfluency annotations
-lm string, Location of where to write a clean language\
model files out of this corpus
-pos boolean, Whether to write a word2pos mapping folder
in the sister directory to the corpusLocation, else assume it is there
-p boolean, whether to include partial words or not
-d boolean, include dialogue act tags in the info
"""
if create_disf_corpus:
    print("Creating corpus...")
    write_pos_map = True
    for div, divfile in file_divisions_transcripts:
        c = [sys.executable,
             THIS_DIR + '/../corpus/disfluency_corpus_creator.py',
             '-i', THIS_DIR + '/../data/raw_data/swda',
             '-t', THIS_DIR + '/../data/disfluency_detection/switchboard',
             '-f', divfile,
             '-a', THIS_DIR +
             '/../data/disfluency_detection/swda_disfluency_annotations',
             # '-lm', "data/lm_corpora",
             '-d'
             ]
        if partial:
            c.append('-p')
        if write_pos_map:
            c.append('-pos')
            write_pos_map = False  # just call it once
        subprocess.call(c)
    print("Finished creating corpus.")

# 2. Run the preprocessing and extraction of features for all files
"""
note to get the audio feature extraction to work you need to have
optional arguments are:
-i string, path of source disfluency corpus
-m string, target path of folder feature matrices in this folder
 (rather than use text files)
-f string, path of file with the division of files to be turned into
-a corpus of vectors
-p boolean, whether to include partial words or not
a string, path to word alignment folder
-tag string, path of folder with tag representations
-new_tag bool, whether to write new tag representations or use old ones
-pos path, path to POS tagger if using one, if None use gold
-train_pos bool, whether to train pos tagger or not and put it in pos
-u bool, include utterance segmentation tags, derivable from utts
-d bool, include dialogue act tags
-l bool, include laughter tags on words- either speech laugh on word or
bout
-joint bool, include big joint tag set as well as the individual ones
-lm string, Location of where to write a clean language\
model files out of this corpus
-xlm boolean, Whether to use a cross language model\
training to be used for getting lm features on the same data.
-asr boolean, whether to produce ASR results for creation of the
data or not
-credentials string, username:password for IBM ASR
-audio string, path to open smile for audio features, if None
no audio extraction"""

if extract_features:
    print("Extracting features...")
    tags_created = False
    tagger_trained = False
    MATRIX_DIR = THIS_DIR + '/../data/disfluency_detection/feature_matrices'
    if not os.path.exists(MATRIX_DIR):
        os.mkdir(MATRIX_DIR)
    for div, div_file in file_divisions_transcripts:
        c = [sys.executable,
             THIS_DIR + '/../feature_extraction/extract_features.py',
             '-i', THIS_DIR + '/../data/disfluency_detection/switchboard',
             '-m', MATRIX_DIR + '/' + div,
             '-f', div_file,
             '-a', THIS_DIR + '/../data/raw_data/swbd_alignments/alignments',
             '-tag', THIS_DIR + '/../data/tag_representations'
             # '-lm', 'data/lm_corpora'
             ]
        if partial:
            c.append('-p')
        if 'train' in div and '-lm' in c:
            c.append('-xlm')
        if not tags_created:
            c.append('-new_tag')
            tags_created = True
            if asr and 'ASR' in div:
                c.extend(['-pos', 'data/crfpostagger'])
                if not tagger_trained:
                    c.append('-train_pos')
                credentials = \
                    '1841487c-30f4-4450-90bd-38d1271df295:EcqA8yIP7HBZ'
                c.extend(['-asr', '-credentials', credentials])
        subprocess.call(c)
    print("Finished extracting features.")

# 3. Train the model on the transcripts (and audio data if available)
# NB each of these experiments can take up to 24 hours
systems_best_epoch = {}
if train_models:
    feature_matrices_filepath = THIS_DIR + '/../data/disfluency_detection/' + \
        'feature_matrices/train'
    validation_filepath = THIS_DIR + '/../data/disfluency_detection/' + \
        'feature_matrices/heldout'
    # train until convergence
    # on the settings according to the numbered experiments in
    # experiments/config.csv file
    for exp in experiments:
        disf = DeepDisfluencyTagger(
            config_file=THIS_DIR + "/experiment_configs.csv",
            config_number=exp
            )
        exp_str = '%03d' % exp
        e = disf.train_net(
                    train_dialogues_filepath=feature_matrices_filepath,
                    validation_dialogues_filepath=validation_filepath,
                    model_dir=THIS_DIR + '/' + exp_str,
                    tag_accuracy_file_path=THIS_DIR +
                    '/results/tag_accuracies/{}.text'.format(exp_str))
        systems_best_epoch[exp] = e
else:
    # Take our word for it that the saved models are the best ones:
    systems_best_epoch[21] = 40
    systems_best_epoch[41] = 16

# 4. Test the models on the test transcripts according to the best epochs
# from training.
# The output from the models is made in the folders
# For now all use timing data
if test_models:
    print("testing models...")
    for exp, best_epoch in sorted(list(systems_best_epoch.items()),
                                  key=lambda x: x[0]):
        exp_str = '%03d' % exp
        # load the model
        disf = DeepDisfluencyTagger(
                        config_file=THIS_DIR + '/experiment_configs.csv',
                        config_number=exp,
                        saved_model_dir=THIS_DIR +
                        '/{0}/epoch_{1}'.format(exp_str, best_epoch)
                                    )
        # simulating (or using real) ASR results
        # for now just saving these in the same folder as the best epoch
        # also outputs the speed
        partial_string = '_partial' if partial else ''
        for div in ["heldout", "test"]:
            disf.incremental_output_from_file(
                    THIS_DIR + '/../data/disfluency_detection/switchboard/' +
                    'swbd_disf_{0}{1}_data_timings.csv'
                    .format(div, partial_string),
                    target_file_path=THIS_DIR + '/{0}/epoch_{1}/'.format(
                        exp_str, best_epoch) +
                    'swbd_disf_{0}{1}_data_output_increco.text'
                    .format(div, partial_string))

# 5. To get the numbers run the notebook:
# experiments/analysis/Interspeech_2015_EMNLP_2015/Interspeech_2015_eval.ipynb
# The results should be consistent with that in the Interspeech 2015 paper.
