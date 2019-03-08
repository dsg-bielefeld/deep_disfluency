import numpy
from copy import deepcopy


def populate_embeddings(emb_dim, vocsize, words2index, pretrained):
    """From pre-trained embeddings and matrix dimension generates a random
    matrix first,
    then assigns the appropriate slices from the pre-learned embeddings.
    The words without a corresponding embedding inherit random init.
    """
    assert pretrained.layer1_size == emb_dim, str(pretrained.layer1_size) + \
        " " + str(emb_dim)
    emb = 0.2 * numpy.random.uniform(-1.0, 1.0,
                                     (vocsize+1, emb_dim)).astype('Float32')
    vocab = deepcopy(list(words2index.keys()))
    for i in range(0, len(pretrained.index2word)):
        word = pretrained.index2word[i]
        index = words2index.get(word)
        if index is None:
            # i.e. no index for this word
            print("no such word in vocab for embedding for word:", word)
            continue
        # print i, word # returns the word and its index in pretrained
        emb[index] = pretrained[word]  # assign the correct index
        vocab.remove(word)
    print(len(vocab), "words with no pretrained embedding.")
    for v in vocab:
        print(v)
    return emb
