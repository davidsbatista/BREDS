import pytest

from breds.seed import Seed


def test_seeds_are_equal():
    seed_one = Seed("DynCorp", "Reston, Va.")

    assert seed_one.ent1 == "DynCorp"
    assert seed_one.ent2 == "Reston, Va."

    seed_two = Seed("DynCorp", "Reston, Va.")

    assert seed_two.ent1 == "DynCorp"
    assert seed_two.ent2 == "Reston, Va."

    assert seed_one == seed_two


def test_seeds_not_implemented():
    seed_one = Seed("DynCorp", "Reston, Va.")

    with pytest.raises(Exception):
        assert seed_one == 4
