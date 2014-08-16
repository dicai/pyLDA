"""
gibbs.py -- implementation of collapsed Gibbs sampler for LDA from
    Finding Scientific Topics, (Griffiths and Steyvers, 2004).

Diana Cai
dcai -at- post dot harvard dot edu

"""
import numpy as np
import time

from parsedocs import preprocess
from lda import LDA

class LDAGibbs(LDA):

    def __init__(self, docs, K, alpha, eta):
        LDA.__init__(self, docs, K, alpha, eta)

        ### Gibbs sampler related data structures ###

        # C_VK[w,k] := number of times word w is assigned to topic k
        self.C_VK = np.zeros((self.V, self.K), dtype=int)
        # C_DK[d,k] := number of times topic k is present in document d
        self.C_DK = np.zeros((self.D, self.K), dtype=int)

        # Cache these values as we go (equivalent to performing column sums for above matrices)
        # For each document, total number of topics assigned
        self.total_topics_per_doc = np.zeros(self.D)
        # For each topic, total number of words assigned to it
        self.total_words_per_topic = np.zeros(self.K)

        # Save results here
        self.log_prob = []
        self.samples = []

    def _conditional_distribution(self, word, d):

        left = (self.C_VK[word, :] + self.eta[word]) / \
                (self.total_words_per_topic + self.eta_sum)

        right = (self.C_DK[d, :] + self.alpha) / \
                (self.total_topics_per_doc[d] + self.alpha_sum)

        dist = left * right

        # Normalize distribution
        dist /= np.sum(dist)

        # for debugging purposes
        self.dist = dist

        return dist

    def _sample_index(self, p):
        """
        Sample from a  Multinomial and return the index

        p: vector of probabilities of length K
        """

        probs = p / np.sum(p)
        draw = np.random.multinomial(1, probs)
        return draw.argmax()

    def print_topics(self):
        pass

    def run_gibbs(self, num_its=5, random_seed=None, verbose=False):
        """
        run the collapsed Gibbs sampler for LDA

        arguments:
            num_its: number of gibbs sweeps to run
            random_seed: set the seed

        Returns a list of the log_evidence at each iteration??

        """

        # initialize topic assignments randomly
        np.random.seed(random_seed)
        print 'Initializing words to random topics... (Iteration 1)'
        time1 = time.time()
        self._gibbs_iteration(init=True)
        time2 = time.time()
        print("Time: %.3f" % (time2-time1))

        if verbose:
            self.print_topics()

        for i in xrange(2, num_its):

            print '\nIteration %d...' % i
            time1 = time.time()
            self._gibbs_iteration(it=i)
            time2 = time.time()
            print("Time: %.3f" % (time2-time1))

            # calculate log evidence
            log_p = self.get_log_probability()
            print 'Log evidence: %s' % log_p
            self.log_prob.append(log_p)

            if verbose:
                self.print_topics()

    def _gibbs_iteration(self, init=False, it=1):

        for d in xrange(self.D):
            doc = self.docs[d]
            for index, (w, topic_assignment) in enumerate(doc):

                # get the word index
                word = self.words[w]

                # If normal iteration, first decrement counts
                if not init:
                    topic = topic_assignment
                    # Decrement counts
                    self.C_VK[word, topic] -= 1
                    self.C_DK[d, topic] -= 1
                    self.total_topics_per_doc[d] -= 1
                    self.total_words_per_topic[topic] -= 1

                # Get distribution -- if init, this is Uniform
                dist = self._conditional_distribution(word, d)
                # Sample topic from this distribution
                topic = self._sample_index(dist)
                # set topic assignment variable
                self.docs[d][index][1] = topic

                # Update counts
                self.C_VK[word, topic] += 1
                self.C_DK[d, topic] += 1
                # Update total counts
                self.total_topics_per_doc[d] += 1
                self.total_words_per_topic[topic] += 1

    def get_log_probability(self):

        C_VK, C_DK = self.C_VK, self.C_DK
        eta, eta_sum = self.eta, self.eta_sum
        alpha, alpha_sum = self.alpha, self.alpha_sum
        total = self.total_words_per_topic

        lp = 0.0
        C_VK.fill(0)
        C_DK.fill(0)
        total.fill(0)

        for d in xrange(self.D):
            doc = self.docs[d]
            for index, (w, topic) in enumerate(doc):

                word = self.words[w]
                left = C_VK[word, topic] / (total[topic] + eta_sum)
                right = C_DK[d, topic] / (index + alpha_sum)
                lp += np.log(left * right)

                C_VK[word, topic] += 1
                C_DK[d, topic] += 1
                total[topic] += 1

        return lp

    def get_empirical_topics(self):
        """
        Returns a K x V matrix, so each row is topic vector (dist over the vocabulary).
        """
        topics = self.C_VK + self.eta[np.newaxis]
        topics /= self.C_VK.sum(0) + self.eta_sum
        return topics.T

    def get_empirical_topic_props(self):
        """
        Returns a D x K matrix, so each row is a distribution over the topics.
        """
        props = self.C_DK + self.alpha
        props /= self.C_DK.sum(1) + self.alpha_sum
        return props

if __name__=="__main__":

    docs = preprocess('text/nips.txt', 'text/stopwordlist.txt', 'text/otherwords.txt')

    num_test = 100

    # split documents up into test and training sets
    train = docs[:-num_test]
    test = docs[-num_test:]

    lda = LDAGibbs(docs, K=100, alpha=0.1, eta=0.1)
    lda.run_gibbs(num_its=2000)

