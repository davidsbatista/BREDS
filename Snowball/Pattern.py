__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Pattern(object):

    def __init__(self, t=None):
        self.positive = 0
        self.negative = 0
        self.confidence = 0
        self.tuples = set()
        self.centroid_bef = list()
        self.centroid_bet = list()
        self.centroid_aft = list()
        if tuple is not None:
            self.tuples.add(t)
            self.centroid_bef = t.bef_vector
            self.centroid_bet = t.bet_vector
            self.centroid_aft = t.aft_vector

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def __str__(self):
        output = ''
        for t in self.tuples:
            output += str(t)+'\n'
        return output

    def update_confidence(self):
        if self.positive or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.add(t)
        self.centroid(self)

    def update_selectivity(self, t, config):
        for s in config.seed_tuples:
            if s.e1 == t.e1 or s.e1.strip() == t.e1.strip():
                if s.e2 == t.e2.strip() or s.e2.strip() == t.e2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
        self.update_confidence()

    @staticmethod
    def centroid(self):
        print "tuples in cluster", len(self.tuples)
        if len(self.tuples) == 1:
            t = next(iter(self.tuples))
            self.centroid_bef = t.bef_vector
            self.centroid_bet = t.bet_vector
            self.centroid_aft = t.aft_vector
        else:
            # TODO: calculate this centroid"
            print "Calculate centroid"
            for t in self.tuples:
                current_words = [e[0] for e in self.centroid_bef]
                for word in t.bef_vector:
                    if word[0] in current_words:
                        pass
                        # somar valor to tf-idf
                        # actualizar o tuplo (w,tf-idf) com este novo valor
                    else:
                        self.centroid_bef.append(word)
            # dividir o tf-idif de cada (w,tf-idf), pelo numero de tuples


























