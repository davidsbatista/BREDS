__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Pattern(object):

    def __init__(self, t=None):
        self.positive = 0
        self.negative = 0
        self.confidence = 0
        self.tuples = list()
        self.centroid_bef = list()
        self.centroid_bet = list()
        self.centroid_aft = list()
        if tuple is not None:
            self.tuples.append(t)
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
            output += str(t)+'|'
        return output

    def update_confidence(self):
        if self.positive or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.append(t)
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
        # it there just one tuple associated with this pattern
        # centroid is the tuple
        if len(self.tuples) == 1:
            t = next(iter(self.tuples))
            self.centroid_bef = t.bef_vector
            self.centroid_bet = t.bet_vector
            self.centroid_aft = t.aft_vector
        else:
            # if there are more tuples associated, calculate the average over all vectors
            print "Calculating centroid"
            print "Tuples in cluster", len(self.tuples)
            for t in self.tuples:
                print t, t.bet_vector
            print "\n"

            # set first tuple as centroid
            self.centroid_bet = self.tuples[0].bet_vector

            # add all other words from other tuples
            for t in range(1, len(self.tuples), 1):
                current_words = [e[0] for e in self.centroid_bet]
                print "current words:", current_words
                for word in self.tuples[t].bet_vector:
                    # if word already exists in centroid, update its tf-idf
                    if word[0] in current_words:
                        print "word already seen"
                        # get the current tf-idf for this word in the centroid
                        for i in range(0, len(self.centroid_bet), 1):
                            if self.centroid_bet[i][0] == word[0]:
                                current_tf_idf = self.centroid_bet[i][1]
                                # sum the tf-idf from the tuple to the current tf_idf
                                current_tf_idf += word[1]
                                # update (w,tf-idf) in the centroid
                                w_new = list(self.centroid_bet[i])
                                w_new[1] = current_tf_idf
                                self.centroid_bet[i] = tuple(w_new)
                                break
                    # if it is not in the centroid, added it with the associated tf-idf score
                    else:
                        self.centroid_bet.append(word)

            # dividir o tf-idf de cada tuple (w,tf-idf), pelo numero de vectores
            for i in range(0, len(self.centroid_bet), 1):
                tmp = list(self.centroid_bet[i])
                tmp[1] /= len(self.tuples)
                self.centroid_bet[i] = tuple(tmp)

            print "centroid updated"
            print self.centroid_bet