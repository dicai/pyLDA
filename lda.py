import numpy as np
from parsedocs import preprocess

class LDA:

    def __init__(self, docs, K, alpha, eta):

        """
        docs: object containing the following
            vocab: the set of words to include (filters out stopwords, etc.)
            D: number of documents

        K: number of topics
        alpha: (scalar) hyperparameter for topic proportions
        eta: (scalar) hyperparameter for topics
        """
        # Corpus-related values
        self.vocab = docs.vocab
        self.words = docs.words
        self.docs = docs.docs
        self.len_docs = docs.len_docs

        # a list of docs, where each doc is represented by a dictionary of word counts
        self.doc_dicts = docs.doc_dicts

        # LDA-related constants
        self.V = len(self.vocab)
        self.D = docs.D
        self.K = K
        # these are vectors of size K and V
        self.alpha = alpha * np.ones(self.K)
        self.alpha_sum = alpha * self.K
        self.eta = eta * np.ones(self.V)
        self.eta_sum = eta * self.V

#if __name__=="__main__":
#
#    docs = preprocess('data/nips.txt', 'words/stopwordlist.txt', 'words/otherwords.txt')
#    lda = LDA(docs, K=100, alpha=0.1, eta=0.1)
