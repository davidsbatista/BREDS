__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from numpy import zeros


class Word2VecWrapper(object):

    @staticmethod
    def pattern2vector_sum(tokens, config):
        """
        Generate word2vec vectors based on words that mediate the relationship
        which can be ReVerb patterns or the words around the entities
        """
        # sum each word
        pattern_vector = zeros(config.vec_dim)

        if len(tokens) > 1:
            for t in tokens:
                try:
                    vector = config.word2vec[t.strip()]
                    pattern_vector += vector
                except KeyError:
                    continue

        elif len(tokens) == 1:
            try:
                pattern_vector = config.word2vec[tokens[0].strip()]
            except KeyError:
                pass

        return pattern_vector

    @staticmethod
    def pattern2vector_average(tokens, config):
        # average of the embedings
        pattern_vector = zeros(config.vec_dim)

        if len(tokens) > 1:
            for t in tokens:
                try:
                    vector = config.word2vec[t.strip()]
                    pattern_vector += vector
                except KeyError:
                    continue

        elif len(tokens) == 1:
            try:
                pattern_vector = config.word2vec[tokens[0].strip()]
            except KeyError:
                pass

        for i in range(0, len(pattern_vector)):
            pattern_vector[i] /= len(pattern_vector)

        return pattern_vector