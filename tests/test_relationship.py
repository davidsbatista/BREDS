from src.breds.sentence import Sentence, Relationship


def test_relationship():
    sentence = (
        "<LOC>DynCorp</LOC>, headquartered in <LOC>Reston, Va.</LOC> gets 98 percent of its revenue "
        "from government work."
    )

    sent = Sentence(sentence, e1_type="LOC", e2_type="LOC", max_tokens=5, min_tokens=2, window_size=6, pos_tagger=None)
