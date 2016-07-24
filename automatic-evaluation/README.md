Automatic Evaluation of Relation Extraction Systems on Large-scale
===================================================================

This is an implementation of a framework for large-scale evaluation of relation extraction systems based
on an automatic annotator. For more details, please refer to:

- Mirko Bronzi, Zhaochen Guo, Filipe Mesquita, Denilson Barbosa, Paolo Merialdo, [Automatic Evaluation of Relation Extraction Systems on Large-scale](https://akbcwekex2012.files.wordpress.com/2012/05/8_paper.pdf). In Proceedings of the Joint Workshop on Automatic Knowledge Base Construction and Web-scale Knowledge Extraction, AKBC-WEKEX 2012


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
