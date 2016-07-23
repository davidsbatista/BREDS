Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics
======================================================================================

This is the sofware implementation for the algorithm proposed in:

David S Batista, Bruno Martins, and Mário J Silva. , `Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics <http://davidsbatista.github.io/publications/breds-emnlp_15.pdf>`_ . In Empirical Methods in Natural Language Processing. ACL, 2015. (Honorable Mention for Best Short Paper)


Usage:

    BREDS-parallel.py parameters sentences positive_seeds negative_simties similarity confidance #cpus

**parameters**:

A sample configuration is provided in `parameters.cfg`. The file contains values for differentes parameters:

    max_tokens_away=6           # maximum number of tokens between the two entities
    min_tokens_away=1           # maximum number of tokens between the two entities
    context_window_size=2       # number of tokens to the left and right

    wUpdt=0.5                   # < 0.5 trusts new examples less on each iteration
    number_iterations=4         # number of bootstrap iterations
    wUnk=0.1                    # weight given to unknown extracted relationship instances
    wNeg=2                      # weight given to extracted relationship instances
    min_pattern_support=2       # minimum number of instances in a cluster to be considered a pattern

    word2vec_path=vectors.bin   # path to a word2vecmodel in binary format

    alpha=0.2                   # weight of the BEF context in the similarity function
    beta=0.6                    # weight of the BET context in the similarity function
    gamma=0.2                   # weight of the AFT context in the similarity function




**sentences**:

A text file containing sentences, one per line, with tags identifing the named entities, e.g.:
 
    The tech company <ORG>Soundcloud</ORG> is based in <LOC>Berlin</LOC>, capital of Germany.
    <ORG>Pfizer</ORG> says it has hired <ORG>Morgan Stanley</ORG> to conduct the review.
    <ORG>Allianz</ORG>, based in <LOC>Munich</LOC>, said net income rose to EUR 1.32 billion ($1.96 billion).
    <ORG>Pfizer</ORG>, based in <LOC>New York City</LOC> , employs about 90,000 workers.

**positive_seeds**

A file with examples of the relationships to be bootstrapped. The file must also specify the semantic type of the
entities in the relationships. The first two lines specify that first entity in the relationship is of type ORG
and that the second is of type LOC. Then a seed relationship is specified per line, e.g.:

    e1:ORG
    e2:LOC

    Nokia;Espoo
    Pfizer;New York
    Google;Mountain View
    Microsoft;Redmond

**negative_seeds**

The same thing as for positive relationships, but containing seeds that do not represent the relationships to be
bootstrapped.

**similarity**

The threshold similarity real value [0,1] for clustering/extracting instances, e.g.:

    0.6

**confidance_threshold**

The confidence threshold real value [0,1] for an instance to be used as seed, e.g.:

    0.8





Example:

    ./BREDS-parallel.py parameters.cfg set_b_matched.txt seeds/affiliation.txt seeds/affiliation_negative.txt 0.6 0.8 4
