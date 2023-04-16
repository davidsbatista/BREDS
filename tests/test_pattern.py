from unittest import TestCase
from unittest.mock import MagicMock

from breds.pattern import Pattern
from breds.seed import Seed
from breds.tuple import Tuple


class TestPattern(TestCase):
    def setUp(self):
        seed_set = set()
        seed_set.add(Seed("seed_1", "seed_2"))
        seed_set.add(Seed("seed_3", "seed_4"))

        self.config = MagicMock()
        self.config.positive_seed_tuples = seed_set

    def test_update_selectivity(self):
        bef_words = ["dummy"]
        bet_words = ["dummy"]
        aft_words = ["dummy"]

        # positive
        pattern = Pattern()
        t = Tuple("seed_1 ", "seed_2 ", None, bef_words, bet_words, aft_words, self.config)
        pattern.update_selectivity(t, self.config)
        self.assertEqual(pattern.positive, 1)
        self.assertEqual(pattern.negative, 0)
        self.assertEqual(pattern.unknown, 0)

        # negative
        pattern = Pattern()
        t = Tuple("seed_1", "seed_5", None, bef_words, bet_words, aft_words, self.config)
        pattern.update_selectivity(t, self.config)
        self.assertEqual(pattern.negative, 1)
        self.assertEqual(pattern.positive, 0)
        self.assertEqual(pattern.unknown, 0)

        # negative
        pattern = Pattern()
        t = Tuple("seed_1", "seed_3", None, bef_words, bet_words, aft_words, self.config)
        pattern.update_selectivity(t, self.config)
        self.assertEqual(pattern.unknown, 0)
        self.assertEqual(pattern.positive, 0)
        self.assertEqual(pattern.negative, 1)

        # unknown
        pattern = Pattern()
        t = Tuple("seed_4", "seed_5", None, bef_words, bet_words, aft_words, self.config)
        pattern.update_selectivity(t, self.config)
        self.assertEqual(pattern.negative, 0)
        self.assertEqual(pattern.positive, 0)
        self.assertEqual(pattern.unknown, 1)
