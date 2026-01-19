import numpy as np
from tqdm import tqdm
import gensim.corpora as corpora

############### preprocessing functions

def get_vocab(tokenized_corpus):
    id2word = corpora.Dictionary(tokenized_corpus)
    return id2word

def build_dtm(tokenized_corpus, id2word=None, logdtm=False):
    """
    converts a tokenized corpus to a Document Term Matrix. id2word is a gensim dictionary.
    """
    if id2word is None:
        id2word = corpora.Dictionary(tokenized_corpus)
    else:
        id2word = id2word
    id_corpus = [id2word.doc2bow(document) for document in tokenized_corpus]
    vocab = id2word.token2id
    N = len(id_corpus)
    DTM = np.zeros((N, len(vocab)))
    for i in tqdm(range(N)):
        doc = id_corpus[i]
        for id, count in doc:
            DTM[i, id] = count

    if logdtm:
        DTM = np.log(1+DTM)

    return DTM