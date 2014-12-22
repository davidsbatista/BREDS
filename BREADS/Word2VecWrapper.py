__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from numpy import dot, zeros


class Word2VecWrapper(object):

    @staticmethod
    def pattern2vector(tokens, config):
        """
        Generate word2vec vectors based on words that mediate the relationship
        which can be ReVerb patterns or the words around the entities
        """
        # sum each word
        pattern_vector = zeros(config.vec_dim)

        if len(tokens) > 1:
            if len(tokens) > 0:
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