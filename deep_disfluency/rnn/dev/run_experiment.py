import numpy as np
import time
import sys
import subprocess
import os
import argparse
import random
from copy import deepcopy
import itertools
import theano
import theano.tensor as T
import gensim
from collections import defaultdict
import cPickle
# sys.path.append('../') #path to the src files

from rnn.elman import Elman
from rnn.lstm import LSTM
from utils.tools import shuffle, minibatch, contextwin, contextwinbackwards
from utils.tools import corpus_to_indexed_matrix
from utils.accuracy import save_to_disfeval_file
from deep_disfluency.embeddings.load_embeddings import populate_embeddings
from decoder.hmm import FirstOrderHMM
from load.load import load_data_from_array, load_tags
from load.load import load_increco_data_from_file, load_data_from_timings_file
from load.load import switchboard_data
from experiment_util import process_arguments
from experiment_util import save_predictions_and_quick_eval

theano.config.optimizer = 'None'  # speeds things up marginally

# NB mainly for experiments
full_label2idx = load_tags("../data/tag_representations/swbd1_trp_tags.csv")
full_idx2label = dict((k, v) for v, k in full_label2idx.iteritems())  # fir


def run_experiment(args):
    # make the args into a dict
    s = dict()
    for feat, val in args._get_kwargs():
        s[feat] = val
        # print feat,val
    print s
    # raw_input()

    if s['acoustic']:
        # dimension of acoustic features vector (essentially 7 * 50)
        s['acoustic'] = 350

    # TODO if not on Bender this is quite big so can't load training- switch
    # to true on Bender
    pre_load_training = True

    # NB Don't get the tag sets here directly anymore
    print "loading data and tag sets"
    _, _, _, train_dict = switchboard_data(train_data=s['train_data'],
                                           tags=s['tags'])

    # get relevant dictionaries
    # first half (28) the same as the test
    idx2label = dict((k, v) for v, k in train_dict['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in train_dict['words2idx'].iteritems())
    if not train_dict.get('pos2idx') is None:
        idx2pos = dict((k, v) for v, k in train_dict['pos2idx'].iteritems())

    range_dir = "../data/disfluency_detection/swda_divisions_disfluency_detection/"
    range_files = [
        range_dir + "SWDisfTrainWithAudio_ranges.text",
        # range_dir + "SWDisfHeldout_ranges.text", #for testing locally
        range_dir + "SWDisfHeldout_ranges.text",
        range_dir + "SWDisfTest_ranges.text",
        range_dir + "SWDisfHeldoutASR_ranges.text",
        range_dir + "SWDisfTestASR_ranges.text"
    ]

    # get the filenames for the train/heldout/test division
    # we will shuffle the train dialogues, keep heldout and test non-scrambled:
    data = defaultdict(list)
    divisions = ["heldout", "test", "heldout_asr", "test_asr"]
    if not s['use_saved_model']:
        divisions = ["train", "heldout", "test"]

    for key, rangefile in zip(divisions, range_files):
        file_ranges = [line.strip("\n") for line in open(rangefile)]
        if s['use_saved_model']:
            if "train" in key:
                continue
            if "asr" in key:
                corpus = key[:key.find("_")]
                corpus = corpus[0].upper() + corpus[1:]
                increco_file = "../data/asr_results/SWDisf{}_pos_increco.text".format(
                    corpus)
                data[key] = load_increco_data_from_file(
                    increco_file, train_dict['words2idx'], train_dict['pos2idx'])

            else:
                #corpus = corpus[0].upper() + corpus[1:]
                file = "../data/disfluency_detection/switchboard/swbd_{}_partial_timings_data.csv".format(
                    key)
                data[key] = load_data_from_timings_file(
                    file, train_dict['words2idx'], train_dict['pos2idx'])
                # print filter(lambda x : x[0] == "4649A",[ data[key][i] for i
                # in range(0, len(data[key]))])[0]
            continue
        # if not using saved model load from pickles
        for f in file_ranges:
            for part in ["A", "B"]:
                dialogue_speaker_data = "../data/audio/" + f + part + ".npy"
                # if we're pre-loading up data, which we do anyway for the test/heldout
                # but also for training when using a big machine
                if key != "train" or pre_load_training:
                    print "loading", dialogue_speaker_data
                    dialogue_speaker_data = np.load(dialogue_speaker_data)
                    dialogue_speaker_data = load_data_from_array(dialogue_speaker_data,
                                                                 n_acoust=s['acoustic'],
                                                                 cs=s['window'],
                                                                 tags=s["tags"],
                                                                 full_idx_to_label_dict=full_idx2label,
                                                                 idx_to_label=idx2label)
                # tuple of name + data/file location
                data[key].append((f + part, dialogue_speaker_data))

    vocsize = len(train_dict['words2idx'].items())
    nclasses = len(train_dict['labels2idx'].items())
    #nsentences = len(train_lex)
    possize = None
    if not train_dict.get('pos2idx') is None:
        possize = len(idx2pos.items())
    #nwords = len(list(itertools.chain(*train_y)))

    print str(len(train_dict['labels2idx'].items())) + " training classes"
    print str(len(train_dict['words2idx'].items())) + " words in vocab"
    if not train_dict.get('pos2idx') is None:
        print str(len(train_dict['pos2idx'].items())) + " pos tags in vocab"
    # print str(nsentences) + " training sequences"
    na = 0
    if s['acoustic']:
        print "with acoustic data..."
        na = s['acoustic']

    print "instantiating model " + s['model'] + "..."
    np.random.seed(s['seed'])
    rnn = None
    random.seed(s['seed'])

    if s['model'] == 'elman':
        rnn = Elman(ne=vocsize,
                    de=s['emb_dimension'],
                    nh=s['nhidden'],
                    na=na,
                    n_out=nclasses,
                    cs=s['window'],
                    npos=possize,
                    update_embeddings=s['update_embeddings'])
    elif s['model'] == 'lstm':
        rnn = LSTM(ne=vocsize,
                   de=s['emb_dimension'],
                   n_lstm=s['nhidden'],
                   na=na,
                   n_out=nclasses,
                   cs=s['window'],
                   npos=possize,
                   lr=s['lr'],
                   single_output=True,
                   output_activation=T.nnet.softmax,
                   cost_function='nll')

    if s['decoder_file']:
        print "instantiating hmm decoder..."
        # add the interregnum tag (not predicted by the rnn, derived from
        # context)
        hmm_dict = deepcopy(train_dict['labels2idx'])
        intereg_ind = len(hmm_dict.keys())
        hmm_dict["<i/><cc>"] = intereg_ind  # add the interregnum tag
        print "loading timing model"
        with open('../decoder/LogReg_balanced_timing_classifier.pkl',
                  'rb') as fid:
            timing_model = cPickle.load(fid)
        with open('../decoder/LogReg_balanced_timing_scaler.pkl',
                  'rb') as fid:
            timing_model_scaler = cPickle.load(fid)
        hmm = FirstOrderHMM(hmm_dict, rnn=rnn, markov_model_file=s['tags'],
                            timing_model=timing_model,
                            timing_model_scaler=timing_model_scaler)
    else:
        hmm = None

    if s['embeddings']:
        print "loading embeddings..."
        pretrained = gensim.models.Word2Vec.load(
            "../embeddings/" + s['embeddings'])  # load pre-trained embeddings
        print pretrained[pretrained.index2word[0]].shape
        # print pretrained[0].shape
        emb = populate_embeddings(
            s['emb_dimension'],
            vocsize,
            train_dict['words2idx'],
            pretrained)  # assign and fill in the gaps
        rnn.load_weights(emb)

    # make folder for the whole experiment
    folder = os.path.basename(__file__).split(
        '.')[0].replace("run_experiment", "") + s['exp_id']
    #folder  = "/home/dsg-labuser/Desktop/rnn_experiments/"+ s['exp_id']
    if os.path.exists(folder):
        if not s['use_saved_model']:
            quit = raw_input(
                'Overwrite contents of folder for experiment {}? [y][n]'.format(
                    s['exp_id']))
            if quit != "y":
                return
    else:
        os.mkdir(folder)

    # make results files
    if not s['use_saved_model']:
        resultsFile = open(
            "results/learning_curves/" +
            s['exp_id'] +
            ".csv",
            "w")
        resultsFile.write("epoch,heldout_loss,heldout_class_loss,val_tags_f,heldout_f_rmtto,heldout_frm,heldout_tto1,heldout_tto2,test_loss,test_class_loss,test_tags_f,test_f_rmtto,test_frm,test_tto1,test_tto2,train_loss\n")
        summariesFile = open(
            "results/tag_accuracies/" +
            s['exp_id'] +
            ".text",
            "w")

    # set current learning rate as the initial learning rate
    s['clr'] = s['lr']
    s['best_epoch'] = 0
    best_f1 = -np.inf  # lowest f-score possible

    print "training..."
    start = 1
    end = s['nepochs']
    if s['use_saved_model']:
        # start from the epoch of the stored model
        start = s['use_saved_model']
        end = start
    for e in range(start, end + 1):

        s['current_epoch'] = e
        tic = time.time()

        # make folder to store this epoch
        epochfolder = folder + "/epoch_" + str(e)
        if not os.path.exists(epochfolder) and not s['use_saved_model']:
            os.mkdir(epochfolder)

        if s['use_saved_model']:
            print "loading stored model and weights- not actually training from " + epochfolder
            rnn.load_weights_from_folder(epochfolder)
            # if s['model'] == "lstm":
            #    rnn.load_weights_from_folder(epochfolder)
            # else:
            #    emb, Wx, Wh, W, bh, b, h0  = rnn.load(epochfolder)
            #    rnn.load_weights(emb, Wx, Wh, W, bh, b, h0)
        else:
            if s['pos']:
                train_loss = rnn.fit(
                    data['train'],
                    s['clr'],
                    acoustic=args.acoustic,
                    load_data=pre_load_training == False)
            else:
                pass

        if s['verbose']:  # output final learning time
            print '[learning] epoch %i >>' % (e), 'completed in %.2f (sec) <<\r' % (time.time() - tic),

        print "saving predictions and evaluating tags..."

        results = {}
        for corpus in ['heldout',
                       'test']:  # nb for training just test and heldout
            # if not corpus == 'heldout': continue
            print corpus
            predictions_file = epochfolder + \
                '/predictions_{}.csv'.format(corpus)
            incremental_eval = False
            if s['use_saved_model']:
                predictions_file = epochfolder + \
                    '/predictions_inc_{}.csv'.format(corpus)
                incremental_eval = True
            corpus_results = save_predictions_and_quick_eval(predictions_filename=predictions_file,
                                                             groundtruth_filename=s['{}_file'.format(
                                                                 corpus.replace("_asr", ""))],
                                                             model=rnn,
                                                             hmm=hmm,
                                                             dialogues=data[corpus],
                                                             idx_to_label_dict=idx2label,
                                                             idx_to_word_dict=idx2word,
                                                             incremental_eval=incremental_eval,
                                                             s=s,
                                                             increco_style="_asr" in corpus or incremental_eval)
            for key, val in corpus_results.items():
                results[corpus + '_' + key] = val
        if s['use_saved_model']:
            # for using saved model we assume only one eval on that given epoch
            # we want to evaluate it fully incrementally/do the error anlaysis
            return {}

        # NB for now not bothering with heldout/test loss
        #results['heldout_loss'] = 100.0
        #results['test_loss'] = 100.0

        print "saving epoch folder and writing results to file"
        rnn.save(epochfolder)  # Epoch file dump

        coi = 'rmtto'  # class of interest

        resultsFile.write(str(e) + "," + str(results['heldout_loss']) + "," +
                          str(results['heldout_class_loss']) + "," +
                          str(results['heldout_f1_tags']) + ',' +
                          str(results['heldout_f1_' + coi]) + "," +
                          str(results['heldout_f1_rm']) + "," +
                          str(results['heldout_f1_tto1']) + "," +
                          str(results['heldout_f1_tto2']) + "," +
                          str(results['test_loss']) + "," +
                          str(results['test_class_loss']) + ',' +
                          str(results['test_f1_tags']) + ',' +
                          str(results['test_f1_' + coi]) + "," +
                          str(results['test_f1_rm']) + "," +
                          str(results['test_f1_tto1']) + "," +
                          str(results['test_f1_tto2']) + "," +
                          str(train_loss) + "\n")
        resultsFile.flush()
        summariesFile.write(
            str(e) +
            "\n" +
            results['heldout_tag_summary'] +
            "\n" +
            results['test_tag_summary'] +
            "\n%%%%%%%%%%\n")
        summariesFile.flush()

        # check to see if it beats the current best on the class of interest
        # stopping criterion
        if results['heldout_f1_' + coi] > best_f1:
            rnn.save(folder)
            best_f1 = results['heldout_f1_' + coi]
            if s['verbose']:
                print 'NEW BEST raw labels at epoch ', e, 'best valid', best_f1
                print 'NEW BEST: epoch', e, 'heldout F1', results['heldout_f1_' + coi], 'best test F1', results['test_f1_' + coi], ' ' * 20
            s['vf1'] = results['heldout_f1_' + coi]
            s['tf1'] = results['test_f1_' + coi]
            s['best_epoch'] = e
            # moves this to the main folder, will have the HMM generated predictions
            #subprocess.call(['mv', epochfolder + '/test.txt', folder + '/best_test.txt'])
            #subprocess.call(['mv', epochfolder + '/valid.txt', folder + '/best_valid.txt'])

        # stopping criteria = if no improvement in 10 epochs
        if s['current_epoch'] - s['best_epoch'] >= 10:
            print "stopping, no improvement in 10 epochs"
            break
        # decay
        if s['decay'] and abs(s['best_epoch'] - s['current_epoch']) >= 2:
            # just a steady decay if things aren't improving for 2 epochs, more
            # a hyper param?
            s['clr'] *= 0.85
            print "learning rate decayed, now ", s['clr']
        if s['clr'] < 1e-5:
            print "stopping, below learning rate threshold"
            break
        if s['verbose']:  # output final testing time
            print '[learning] epoch %i >>' % (e), 'testing in %.2f (sec) <<\r' % (time.time() - tic),

    print 'BEST RESULT: epoch', s['best_epoch'], 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder
    resultsFile.close()
    summariesFile.close()


if __name__ == '__main__':
    #allsystemsfinal = [(33,45,'RNN'),(35,6,'LSTM'),(36,15,'LSTM (complex tags)'),(34,37,'RNN (complex tags)')]

    # for exp,e,system in allsystemsfinal:
    #    print exp,e,system
    #    id = "0{0}".format(exp)
    #    args = process_arguments(config='experiment_configs.csv',exp_id=id,use_saved=e,hmm=True)
    #    run_experiment(args)
    #args = process_arguments(config='experiment_configs.csv', exp_id='037', hmm=False)
    # run_experiment(args)
    parser = argparse.ArgumentParser(description='Training regime for\
    RNN/LSTM models.')
    parser.add_argument('-config', action='store', dest='config',
                        default='experiment_configs.csv',
                        help='Location of the config file.')
    parser.add_argument('-exp', action='store', dest='exp_id',
                        default='038',
                        help='Experiment in the config file.')
    parser.add_argument('-hmm', action='store_true', dest='hmm',
                        default=False,
                        help='Whether to use an hmm model or not.')
    parser.add_argument('-epoch', action='store_true', dest='test_epoch',
                        default=None,
                        help='Use saved epoch model.')
    args = parser.parse_args()
    exp_args = process_arguments(config=args.config,
                                 exp_id=args.exp_id,
                                 decoder_file=args.hmm,
                                 use_saved=args.test_epoch)
    run_experiment(exp_args)

    #args = process_arguments(config='experiment_configs.csv', exp_id='039', hmm=False)
    # run_experiment(args)

    # for exp in range(31,40):
    #    args.exp = exp
    # run_experiment(args) #could in theory try multiple instances for
    # multiple experiments
