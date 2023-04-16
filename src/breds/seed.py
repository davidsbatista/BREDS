__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"


class Seed(object):
    def __init__(self, e1: str, e2: str) -> None:
        self.e1 = e1
        self.e2 = e2

    def __hash__(self) -> int:
        return hash(self.e1) ^ hash(self.e2)

    def __eq__(self, other) -> bool:
        return self.e1 == other.e1 and self.e2 == other.e2
