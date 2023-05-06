__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import uuid
from typing import Any, Optional, Set

from breds.breds_tuple import BREDSTuple
from breds.config import Config


class Pattern:  # pylint: disable=too-many-instance-attributes
    """
    A pattern is a set of tuples that is used to extract relationships between named-entities.
    """

    def __init__(self, tpl: Optional[BREDSTuple] = None):
        self.uuid = uuid.uuid4()
        self.positive = 0
        self.negative = 0
        self.unknown = 0
        self.confidence: float = 0.0
        self.tuples = set()
        self.bet_uniques_vectors: Set[Any] = set()
        self.bet_uniques_words: Set[Any] = set()
        if tpl is not None:
            self.tuples.add(tpl)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.tuples == other.tuples

    def __cmp__(self, other: object) -> int:
        """Compare two patterns based on their confidence"""
        if not isinstance(other, Pattern):
            return NotImplemented
        if other.confidence > self.confidence:
            return -1
        if other.confidence < self.confidence:
            return 1
        return 0

    def update_confidence(self, config: Config) -> None:
        """Update the confidence of the pattern"""
        if self.positive > 0:
            self.confidence = float(self.positive) / float(
                self.positive + self.unknown * config.w_unk + self.negative * config.w_neg
            )
        elif self.positive == 0:
            self.confidence = 0

    def add_tuple(self, tpl: BREDSTuple) -> None:
        """Add another tuple to be used to generate the pattern"""
        self.tuples.add(tpl)

    def merge_all_tuples_bet(self) -> None:
        """Put all tuples with BET vectors into a set so that comparison with repeated vectors is eliminated"""
        self.bet_uniques_vectors = set()
        self.bet_uniques_words = set()
        for tpl in self.tuples:
            # transform numpy array into a tuple, so it can be hashed and added into a set
            self.bet_uniques_vectors.add(tuple(tpl.bet_vector))  # type: ignore
            self.bet_uniques_words.add(tpl.bet_words)

    def update_selectivity(self, tpl: BREDSTuple, config: Config) -> None:
        """Update the selectivity of the pattern"""
        matched_both = False
        matched_e1 = False

        for seed in config.positive_seed_tuples:
            if seed.ent1.strip() == tpl.ent1.strip():
                matched_e1 = True
                if seed.ent2.strip() == tpl.ent2.strip():
                    self.positive += 1
                    matched_both = True
                    break

        if matched_e1 is True and matched_both is False:
            self.negative += 1

        if matched_both is False:
            for ngt_seed in config.negative_seed_tuples:
                if ngt_seed.ent1.strip() == tpl.ent1.strip():
                    if ngt_seed.ent2.strip() == tpl.ent2.strip():
                        self.negative += 1
                        matched_both = True
                        break

        if not matched_both and not matched_e1:
            self.unknown += 1
