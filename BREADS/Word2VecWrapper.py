__author__ = 'dsbatista'

import numpy as np


class Word2VecWrapper(object):

    @staticmethod
    def pattern2vector(tokens, config):
        """
        Generate word2vec vectors based on words that mediate the relationship
        which can be ReVerb patterns or the words around the entities
        """
        # sum each word
        if len(tokens) > 0:
            pattern_vector = np.zeros(config.vec_dim)
            for t in tokens:
                try:
                    vector = config.word2vec[t.strip()]
                    pattern_vector += vector
                except KeyError:
                    continue

        return pattern_vector