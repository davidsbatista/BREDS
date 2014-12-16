import fileinput

__author__ = 'dsbatista'


class Breads(object):

    @staticmethod
    def generate_tuples(sentences_file):
        """
        Generate tuples instances from a txt file with sentences
        """
        for line in fileinput.input(sentences_file):
            print line

    @staticmethod
    def calculate_tuple_confidence(self):
        """
        Calculates the confidence of a tuple is: Conf(P_i) * DegreeMatch(P_i)
        """
        pass

    @staticmethod
    def compare_pattern_tuples(self):
        """
        Compute similarity of a tuple with all the extraction patterns
        Compare the relational words from the sentence with every extraction pattern
        """
        pass

    @staticmethod
    def expand_patterns(self):
        """
        Expands the relationship words of a pattern based on similarities
        - For each word part of a pattern
            - Construct a set with all the similar words according to Word2Vec given a threshold t
        - Calculate the intersection of all sets
        """
        pass

    @staticmethod
    def iteration(self):
        """
        starts a bootstrap iteration
        """

    @staticmethod
    def match_seeds_tuples(self):
        """
        checks if an extracted tuple matches seeds tuples
        """

    @staticmethod
    def  pattern_drifts(self):
        """
        detects if the patterns drift
        """
