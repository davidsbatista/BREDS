__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

from typing import List, Tuple

from numpy import zeros
from config import Config


class BREDSTuple:  # pylint: disable=too-many-instance-attributes,too-many-arguments
    """
    A Tuple holds the information about a relation between two entities, namely:

    e1: the first entity
    e2: the second entity
    sentence: the sentence where the relation was found
    before: the tokens before the relation
    between: the tokens between the relation
    after: the tokens after the relation
    """

    # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
    filter_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"]

    def __init__(
        self,
        ent1: str,
        ent2: str,
        sentence: str,
        before: List[Tuple[str, str]],
        between: List[Tuple[str, str]],
        after: List[Tuple[str, str]],
        config: Config,  # type: ignore
    ):
        self.ent1 = ent1
        self.ent2 = ent2
        self.sentence = sentence
        self.confidence = 0
        self.bef_tags = before
        self.bet_tags = between
        self.bet_filtered = None
        self.aft_tags = after
        self.bef_words = " ".join([x[0] for x in self.bef_tags])
        self.bet_words = " ".join([x[0] for x in self.bet_tags])
        self.aft_words = " ".join([x[0] for x in self.aft_tags])
        self.bef_vector = None
        self.bet_vector = None
        self.aft_vector = None
        self.passive_voice = False
        self.construct_vectors(config)

    def __str__(self) -> str:
        return str(self.ent1 + "\t" + self.ent2 + "\t" + self.bef_words + "\t" + self.bet_words + "\t" + self.aft_words)

    def __hash__(self) -> int:
        return hash(self.ent1) ^ hash(self.ent2) ^ hash(self.bef_words) ^ hash(self.bet_words) ^ hash(self.aft_words)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BREDSTuple):
            return NotImplemented
        return (
            self.ent1 == other.ent1
            and self.ent2 == other.ent2
            and self.bef_words == other.bef_words
            and self.bet_words == other.bet_words
            and self.aft_words == other.aft_words
        )

    def __cmp__(self, other: object) -> int:
        if not isinstance(other, BREDSTuple):
            return NotImplemented
        if other.confidence > self.confidence:
            return -1
        if other.confidence < self.confidence:
            return 1
        return 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, BREDSTuple):
            return NotImplemented
        return self.confidence < other.confidence

    def construct_vectors(self, config: Config) -> None:
        """
        Construct the vectors for the tuple, based on the words before, between and after the relation.
        """
        # Check if BET context contains a ReVerb pattern
        reverb_pattern = config.reverb.extract_reverb_patterns_tagged_ptb(self.bet_tags)
        if len(reverb_pattern) > 0:
            self.passive_voice = config.reverb.detect_passive_voice(reverb_pattern)
            bet_words = reverb_pattern
        else:
            self.passive_voice = False
            bet_words = self.bet_tags

        self.bet_filtered = [
            t[0] for t in bet_words if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos
        ]

        # compute the vector over the filtered BET context
        self.bet_vector = self.pattern2vector_sum(self.bet_filtered, config)

        # compute the vector for words before the first entity, and for words after the second entity
        bef_no_tags = [t[0] for t in self.bef_tags]
        aft_no_tags = [t[0] for t in self.aft_tags]
        self.bef_vector = self.pattern2vector_sum(bef_no_tags, config)
        self.aft_vector = self.pattern2vector_sum(aft_no_tags, config)

    @staticmethod
    def pattern2vector_sum(tokens: List[str], config: Config):
        """
        Compute the vector for a given pattern, by summing the vectors of the words in the pattern.
        """
        pattern_vector = zeros(config.vec_dim)
        if len(tokens) > 1:
            for tok in tokens:
                try:
                    vector = config.word2vec[tok.strip()]
                    pattern_vector += vector
                except KeyError:
                    continue
        elif len(tokens) == 1:
            try:
                pattern_vector = config.word2vec[tokens[0].strip()]
            except KeyError:
                pass

        return pattern_vector
