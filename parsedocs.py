import re
import numpy as np

min_threshold = 10
max_threshold = 8000
min_letters = 2

def create_stopword_set(filename):
    """
    input: a filename containing stopwords, one word per line
    output: Returns a set of lowercase stopwords.
    """

    f = open(filename)
    stopwords = set(word.strip().lower() for word in f)
    f.close()

    return stopwords

def tokenize(string, stopwords=set()):

    """
    input:
        string -- the string to tokenize
        stopwords -- set of stopwords to remove

    Returns a list of lowercase tokens with the stopwords removed
    """

    tokens = re.findall('[a-z]+', string.lower())

    return [x for x in tokens if x not in stopwords]

def delete_word(word_list, key):
    """
    Deletes a word (key) from a dictionary
    """
    new_list = word_list
    del new_list[key]
    return new_list

def preprocess(filename, stopwords_filename=None, otherwords_filename=None,
        gibbs=True):

    """
    Returns a tuple containing

    input:
        filename -- name of file containing documents. One document per line?
        stopwords_filename -- name of file containing stopwords
        otherwords_filename -- name of file containing other words to remove e.g. rare and common words
    """

    if stopwords_filename:
        stopwords = create_stopword_set(stopwords_filename)
    else:
        print('Stopwords not loaded properly!')
        stopwords = set()

    if otherwords_filename:
        stopwords.update(create_stopword_set(otherwords_filename))

    # read in all the documents in the collection
    print 'Reading in documents...'
    f = open(filename, 'r')
    docs = Docs()
    lines = f.readlines()
    counts = {}
    for d in range(len(lines)):
        doc = lines[d]
        tokens = tokenize(doc, stopwords)
        counts = docs.add(tokens, counts)

    f.close()

    print 'Preprocessing...'

    if gibbs:
        docs.finalize()

    return docs

class Docs:

    def __init__(self, tokens=[], D=0, doc_dicts = [], words={}, docs=[], vocab={}):
        self.vocab = vocab
        # a list of docs, each doc is represented by a list of tuples (word: topic) in order
        self.docs = docs
        self.len_docs = []
        self.D = D

        # word counts of the vocabulary
        if tokens:
            for w in tokens:
                val = self.vocab.get(w, 1) + 1
                self.vocab[w] = val

        # initalize some additional structures for bookkeeping purposes
        # maps each word to an index (sorted alphabetically)
        self.words = words
        self.doc_dicts = doc_dicts


    def __getitem__(self, i):

        return self.doc_dicts[i]

    def __getslice__(self, i, j):

        return Docs(doc_dicts=self.doc_dicts[i:j], words=self.words, vocab=self.vocab, D=self.D)

    def __len__(self):
        return len(self.doc_dicts)


    def add(self, tokens, counts):
        """
            arguments:
                tokens: word tokens we want to add
                counts: dictionary of counts (helps us get rid of low counts)
        """
        # add to the vocabulary
        new_tokens = []
        for w in tokens:
            # require that words are at least of length min_letters
            if len(w) <=  min_letters:
                continue

            counts[w] = counts.get(w, 0)
            # get rid of low count words
            count = counts[w]
            if count <= min_threshold:
                counts[w] += 1
                continue

            self.vocab[w] = self.vocab.get(w, 0) + 1
            new_tokens.append(w)

        # increment number of docs and add their counts
        num_tokens = len(new_tokens)
        new_doc = [list(x) for x in zip(new_tokens, np.zeros(num_tokens))]
        self.docs.append(new_doc)
        self.len_docs.append(num_tokens)
        self.D += 1

        return counts

    def finalize(self):
        """
        Make some additional structures after we finalize our vocabulary set (after pre-processing)
        """

        for index, word in enumerate(sorted(self.vocab.keys())):
            self.words[word] = index

        for d in range(self.D):
            doc = self.docs[d]
            doc_dict = {}
            for (w, t) in doc:
                wordid = self.words[w]
                doc_dict[wordid] = doc_dict.get(wordid, 0) + 1
            self.doc_dicts.append(doc_dict)


    def test_token_condition(self, token):
        """
        Test to see if whether or not we want to keep the token in our vocabulary.

        token: a string for the word we want to test
        """
        if self.vocab[token] <= min_threshold or \
            self.vocab[token] >= max_threshold:

            return True

        else:
            return False

    def remove_word(self, token):
        """
        Remove token from our document collections.

        token: a string for the word we want to remove
        """

        # remove word from our vocabulary
        del self.vocab[token]


def nips():
    f = open('words/nips_vocab2.txt', 'w')

    v = preprocess('data/nips.txt', 'words/stopwordlist.txt', 'words/otherwords.txt')
    a = sorted(v.vocab.keys())

    for w in a:
        f.write(w)
        f.write('\n')

    f.close()

def yelp():
    f = open('words/yelp_vocab.txt', 'w')

    v = preprocess('data/yelp.txt', 'words/stopwordlist.txt', 'words/otherwords.txt')
    a = sorted(v.vocab.keys())

    for w in a:
        f.write(w)
        f.write('\n')

    f.close()

if __name__=="__main__":
    nips()
