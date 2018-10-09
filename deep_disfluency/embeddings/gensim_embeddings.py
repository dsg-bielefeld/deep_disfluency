import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
switchboard_clean_directory = "../data/lm_corpora"


class MySentences(object):
    # a memory-friendly iterator
    def __init__(self, training_files):
        self.training_files = training_files

    def __iter__(self):
        for fname in training_files:
            print fname
            for line in open(fname):
                if "clean" in fname:
                    ID, text = line.split(",")
                else:
                    ID = ""
                    text = line
                if ID == "POS":
                    continue
                yield text.split()


training_files = [switchboard_clean_directory +
                  "/swbd_disf_train_1_2_clean.text"]
sentences = MySentences(training_files)
emb_size = 50
model = gensim.models.Word2Vec(sentences, min_count=2, size=emb_size)
model.save('swbd_clean_50')
print model.index2word[0]
print len(model.index2word)
# print model['um']

# for i in model.index2word:
#     print i
