# -*- coding: utf-8 -*-



def swbd_folds_disfluency_corpus(corpus_input, num_files=496, num_folds=10):
    """Returns num_folds fold division of the input swbd PTB 
    disfluency corpus in num_folds strings
    of the configuration of the division and the folds themselves.

    Keyword Arguments:
    corpus_input -- the (ID,.. up to n features) tuple which will be divded 
    into num_fold tuples
    of the same type
    """
    # we have 10 divisions of 496 (?) files, the smallheldout corpus already
    # there so 9 to start with in rest
    config = []
    index = 0
    # split main clean corpus into 9, have heldout as the other one, and rotate
    folds = []

    # calculate the dividing points based on nearest int
    divs = []
    # goes up to n-1th fold to get the split point
    for i in range(1, num_folds):
        split_point = int((i / num_folds) * num_files)
        divs.append(split_point)
    divs.append(num_files - 1)  # add the last one

    line = input.readline()  # first line
    for d in divs:
        subcorpus = ""
        posSubcorpus = ""
        targetstop = d
        currentSection = line.split(",")[0].split(":")[0]
        current = currentSection
        ranges = []
        print(currentSection)
        while index <= targetstop:
            ranges.append(current)
            while current == currentSection:
                subcorpus += line.split(",")[1] + "\n"
                posSubcorpus += input.readline().split(",")[1] + "\n"

                line = input.readline()  # read the next text level
                if not line:
                    break  # end of file, break to increment index?
                current = line.split(",")[0].split(":")[0]
            currentSection = current
            index += 1
        folds.append((tuple(ranges), subcorpus, posSubcorpus))
        # fold always has structure (ranges,wordsubcorpus(big
        # string),posSubcorpus(big string))

    print("no of folds = ", str(len(folds)))
    for i in range(0, len(folds)):
        test = i
        if i == len(folds) - 1:
            heldout = i - 1
        else:
            heldout = i + 1
        training = []
        for index in range(0, len(folds)):
            if not index == heldout and not index == test:
                training.append(index)  # just appends an index
        # config is always (list of training indices),heldoutout index,
        # test(i.e. where we're assigning probs to)
        config.append((tuple(training), heldout, test))
    print("config size", str(len(config)))
    input.close()
    return config, folds


def bnc_folds():
    source = open("../data/bnc_spoken/bnc_spokenREF.text")
    fold = open("../data/bnc_spoken/bnc_spokenAllREF.text", "w")

    count = 0
    uttcount = 0
    for line in source:
        ref, utt = line.split(",")
        if ref == "":
            pos = False
        else:
            pos = True

        # Could ignore numbers as they are different in the BNC
        # if pos == False:
            #number = False
            # for i in range(0,10):
            #    if str(i) in utt or "-" in utt:
            #        number = True
            #        for line in bnc:
            #            pass
            #            break
            #        break
            # if number == True: continue
        if len(line.split()) == 0:

            if pos == True:
                possline = line
            else:
                normalline = line
        else:
            continue
        if pos == True:
            if not len(possline.split()) == len(normalline.split()):
                print(possline, normalline)
                continue
            # Could ignore filled pauses, as these are different, 
            #though could do normalization
            # if "UH" in possline:
            #    #might be safer to ignore these
            #    continue
            #    #print normalline, possline
            fold.write(normalline)
            fold.write(possline)
            count += len(utt.split())
            uttcount += 1
        # else:
            # if count > 3000000: break
    print(count)
    print(uttcount)

    fold.close()
    source.close()
