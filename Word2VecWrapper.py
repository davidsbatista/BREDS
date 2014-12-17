__author__ = 'dsbatista'

from gensim.models import Word2Vec
import numpy as np


class Word2VecWrapper(object):

    def __init__(self, Word2VecModelPath):
        print "Loading word2vec model ...\n"
        self.model = Word2Vec.load_word2vec_format(Word2VecModelPath, binary=True)

    """
    Generate word2vec vectors based on words that mediate the relationship
    - If between context is empty
    def pattern2vector(tuple):
        # sum each word of a pattern in the 'between' context
        if len(rel.patterns_bet_norm)>0:
            pattern = rel.patterns_bet_norm[0]
            pattern_vector = np.zeros(vectors_dim)
            for word in word_tokenize(pattern):
                try:
                    vector = model[word.strip()]
                    pattern_vector += vector
                except Exception, e:
                    words_not_found.append(word.strip())
            rel.patterns_bet_vectors.append(pattern_vector)
    """