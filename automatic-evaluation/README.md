# "Automatic Evaluation of Relation Extraction Systems on Large-scale"
    # https://akbcwekex2012.files.wordpress.com/2012/05/8_paper.pdf
    #
    # S  - system output
    # D  - database (freebase)
    # G  - will be the resulting ground truth
    # G' - superset, contains true facts, and wrong facts
    # a  - contains correct facts from the system output
    #
    # b  - intersection between the system output and the
    #      database (i.e., freebase),
    #      it is assumed that every fact in this region is correct
    # c  - contains the database facts described in the corpus
    #      but not extracted by the system
    # d  - contains the facts described in the corpus that are not
    #      in the system output nor in the database
    #
    # Precision = |a|+|b| / |S|
    # Recall    = |a|+|b| / |a| + |b| + |c| + |d|
    # F1        = 2*P*R / P+R
