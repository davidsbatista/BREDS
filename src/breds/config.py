import fileinput
import re

from gensim.models import KeyedVectors
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from reverb import Reverb
from seed import Seed

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"


class Config(object):
    """
    Initializes a configuration object with the parameters from the config file:

    - Reads the positive and negative seeds from the files.
    - Reads the word2vec model.
    - Initializes the lemmatizer and the stopwords list.
    - Set the weights for the unknown and negative instances.
    - Set the POS tags to be filtered out.
        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        # select everything except stopwords, ADJ and ADV
    - Set the regex to clean the text.
    - Set the threshold for the similarity between the patterns and the instances.
    - Set the threshold for the confidence of the patterns.
    - Initialize the Reverb object.
    """

    def __init__(self, config_file, positive_seeds, negative_seeds, similarity, confidence) -> None:  # noqa: C901
        self.context_window_size: int = 2
        self.min_tokens_away: int = 1
        self.max_tokens_away: int = 6
        self.similarity = None
        self.alpha = None
        self.word2vec_model_path = None
        self.beta = None
        self.gamma = None
        self.min_pattern_support = None
        self.number_iterations = None
        self.wNeg = None
        self.tag_type = None
        self.wUnk = None
        self.wUpdt = None
        self.filter_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"]
        self.regex_clean_simple = re.compile("</?[A-Z]+>", re.U)
        self.regex_clean_linked = re.compile("</[A-Z]+>|<[A-Z]+ url=[^>]+>", re.U)
        self.tags_regex = re.compile("</?[A-Z]+>", re.U)
        self.positive_seed_tuples = set()
        self.negative_seed_tuples = set()
        self.vec_dim = 0
        self.e1_type = None
        self.e2_type = None
        self.stopwords = stopwords.words("english")
        self.lmtzr = WordNetLemmatizer()
        self.threshold_similarity = similarity
        self.instance_confidence = confidence
        self.reverb = Reverb()
        self.word2vec = None
        self.vec_dim = None
        self.read_config(config_file)
        self.read_seeds(positive_seeds, self.positive_seed_tuples)
        self.read_seeds(negative_seeds, self.negative_seed_tuples)

        print("Configuration parameters")
        print("========================\n")

        print("Relationship/Sentence Representation")
        print("e1 type              :", self.e1_type)
        print("e2 type              :", self.e2_type)
        print("tags type            :", self.tag_type)
        print("context window       :", self.context_window_size)
        print("max tokens away      :", self.max_tokens_away)
        print("min tokens away      :", self.min_tokens_away)
        print("Word2Vec Model       :", self.word2vec_model_path)

        print("\nContext Weighting")
        print("alpha                :", self.alpha)
        print("beta                 :", self.beta)
        print("gamma                :", self.gamma)

        print("\nSeeds")
        print("positive seeds       :", len(self.positive_seed_tuples))
        print("negative seeds       :", len(self.negative_seed_tuples))
        print("negative seeds wNeg  :", self.wNeg)
        print("unknown seeds wUnk   :", self.wUnk)

        print("\nParameters and Thresholds")
        print("threshold_similarity :", self.threshold_similarity)
        print("instance confidence  :", self.instance_confidence)
        print("min_pattern_support  :", self.min_pattern_support)
        print("iterations           :", self.number_iterations)
        print("iteration wUpdt      :", self.wUpdt)
        print("\n")

    def read_config(self, config_file):  # noqa: C901
        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.wUnk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.wNeg = float(line.split("=")[1])

            if line.startswith("number_iterations"):
                self.number_iterations = int(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("similarity"):
                self.similarity = line.split("=")[1].strip()

            if line.startswith("word2vec_path"):
                self.word2vec_model_path = line.split("=")[1].strip()

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

            if line.startswith("tags_type"):
                self.tag_type = line.split("=")[1].strip()

        fileinput.close()
        assert self.alpha + self.beta + self.gamma == 1

    def read_word2vec(self):
        print("Loading word2vec model ...\n")
        self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_model_path, binary=True)
        self.vec_dim = self.word2vec.vector_size
        print(self.vec_dim, "dimensions")

    def read_seeds(self, seeds_file, holder):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                holder.add(seed)
