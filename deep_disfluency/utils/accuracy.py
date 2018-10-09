# Legacy:
def save_to_disfeval_file(p, g, w, f, filename, incremental=False):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words
    f :: original input gold standard file

    OUTPUT:
    filename :: name of the file where the predictions
    are written. In right format for disfluency evaluation
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    if incremental == False:
        for sl, sp, sw in zip(g, p, w):
            out += 'BOS O O\n'
            for wl, wp, w in zip(sl, sp, sw):
                out += w + ' ' + wl + ' ' + wp + '\n'
            out += 'EOS O O\n\n'
    else:
        # We want a less straight forward output- increco style first increment always
        # has a start symbol and the first tag
        # last one always has the end of utt tag and this may be different from the penultimate one which covers the same
        # words, but by virtue of knowing it's the end of the sequence it could change
        # always have an iteration over the ground truth utt to give the prefixes
        # with the predictions all the prefixes of this
        for sl, sp, sw in zip(g, p, w):  # for each utterance
            prefix = []  # init the prefix, the word and the ground truth
            sw.append('EOS')  # adding an extra word position
            sl.append('O')
            sp[-1].append('O')  # trivially not evaluated
            for wl, pp, w in zip(
                    sl, sp, sw):  # for each prefix in the utt pp = prefix prediciton, not just latest word
                prefix.append(w + ' ' + wl + ' ')
                assert(len(prefix) == len(pp)), str(prefix) + str(pp)
                out += 'BOS O O\n'
                for my_prefix, my_prediction in zip(
                        prefix, pp):  # for each prediction
                    out += my_prefix + " " + my_prediction + "\n"
                # last one is final for the prefix, the last one will have an
                # EOS
                out += "\n"

    f = open(filename, 'w')
    f.writelines(out)
    f.close()
    return filename
