import pytest
from breds.config import Config
from breds.breds_tuple import BREDSTuple


@pytest.fixture
def config():
    return Config()


def test_construct_vectors(config):
    e1 = "Nokia"
    e2 = "Espoo"
    sentence = "Nokia is based in Espoo."
    before = []
    between = [("is", "DT"), ("based", "NN"), ("in", "NN")]
    after = []
    breds_tuple = BREDSTuple(e1, e2, sentence, before, between, after, config)

    assert breds_tuple.bef_vector is not None
    assert breds_tuple.bet_vector is not None
    assert breds_tuple.aft_vector is not None
