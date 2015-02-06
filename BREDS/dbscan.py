import random

import numpy as np

# scikit
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise


def generate_vectors():
    v = []
    for i in range(0, 5):
        random.seed()
        v.append(random.random()*200)
        #v.append(random.randint(0,5))
    return v


def main():
    vectors = []
    for i in range(0, 25):
        v = generate_vectors()
        vectors.append(v)
        print v

    n_array = np.array(vectors)

    print "distances:"
    for e in n_array:
        print e
        print "\n"

    matrix = pairwise.pairwise_distances(np.array(vectors), metric='cosine')
    db = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
    db.fit(matrix)

    print "\n"
    for v in range(0, len(vectors)-1):
        print vectors[v], db.labels_[v]


if __name__ == "__main__":
    main()