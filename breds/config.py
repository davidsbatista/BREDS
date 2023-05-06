__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import fileinput
import re
from typing import Any, Optional, Set

from gensim.models import KeyedVectors
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from breds.reverb import Reverb
from breds.seed import Seed


class Config:  # pylint: disable=too-many-instance-attributes, too-many-arguments
    """
    Initializes a configuration object with the parameters from the config file:

    - Reads the positive and negative seeds from the files.
    - Reads the word2vec model.
    - Initializes the lemmatizer and the stopwords list.
    - Set the weights for the unknown and negative instances.
    - Set the POS tags to be filtered out.
        http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        select everything except stopwords, ADJ and ADV
    - Set the regex to clean the text.
    - Set the threshold for the similarity between the patterns and the instances.
    - Set the threshold for the confidence of the patterns.
    - Initialize the Reverb object.
    """

    def __init__(
        self,
        config_file: Optional[str],
        word2vec_model_path: str,
        positive_seeds: str,
        negative_seeds: str,
        similarity: float,
        confidence: float,
        number_iterations: int,
    ) -> None:  # noqa: C901
        if config_file is None:
            self.context_window_size: int = 2
            self.min_tokens_away: int = 1
            self.max_tokens_away: int = 6
            self.similarity: float = 0.6
            self.alpha: float = 0.0
            self.beta: float = 1.0
            self.gamma: float = 0.0
            self.min_pattern_support: int = 4
            self.w_neg: float = 2
            self.w_unk: float = 0.0
            self.w_updt: float = 0.5
        else:
            self.read_config(config_file)

        self.filter_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"]
        self.stopwords = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()
        self.regex_clean_simple = re.compile("</?[A-Z]+>", re.U)
        self.tags_regex = re.compile("</?[A-Z]+>", re.U)
        self.positive_seed_tuples: Set[Any] = set()
        self.negative_seed_tuples: Set[Any] = set()
        self.e1_type: str
        self.e2_type: str
        self.threshold_similarity = similarity
        self.instance_confidence = confidence
        self.number_iterations = number_iterations
        self.word2vec_model_path = word2vec_model_path
        self.reverb = Reverb()
        self.word2vec: Any
        self.vec_dim: int
        self.read_seeds(positive_seeds, self.positive_seed_tuples)
        if negative_seeds:
            self.read_seeds(negative_seeds, self.negative_seed_tuples)

    def print_config(self) -> None:  # pragma: no cover
        # pylint: disable=expression-not-assigned
        """
        Prints the configuration parameters.
        """
        print("Configuration parameters")
        print("========================\n")
        print("e1 type              :", self.e1_type)
        print("e2 type              :", self.e2_type)
        print("context window       :", self.context_window_size)
        print("max tokens away      :", self.max_tokens_away)
        print("min tokens away      :", self.min_tokens_away)
        print("word2vec model       :", self.word2vec_model_path)
        print("\n")
        print("alpha                :", self.alpha)
        print("beta                 :", self.beta)
        print("gamma                :", self.gamma)
        print("\n")
        print("positive seeds       :", len(self.positive_seed_tuples))
        print("negative seeds       :", len(self.negative_seed_tuples)) if self.negative_seed_tuples else "None"
        print("negative seeds wNeg  :", self.w_neg)
        print("unknown seeds wUnk   :", self.w_unk)
        print("\n")
        print("threshold_similarity :", self.threshold_similarity)
        print("instance confidence  :", self.instance_confidence)
        print("min_pattern_support  :", self.min_pattern_support)
        print("iterations           :", self.number_iterations)
        print("iteration wUpdt      :", self.w_updt)
        print("\n")

    def read_config(self, config_file: str) -> None:  # noqa: C901
        # pylint: disable=too-many-branches
        """
        Reads the configuration file and sets the parameters.
        """

        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.w_updt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.w_unk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.w_neg = float(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("similarity"):
                self.similarity = float(line.split("=")[1].strip())

            if line.startswith("word2vec_path"):
                self.word2vec_model_path = line.split("=")[1].strip()

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

        fileinput.close()
        if self.alpha + self.beta + self.gamma != 1:
            raise (ValueError(print("alpha + beta + gamma != 1")))

    def read_word2vec(self, path: str) -> KeyedVectors:  # type: ignore
        """Reads the word2vec model."""

        print("Loading word2vec model ...\n")
        word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
        self.vec_dim = word2vec.vector_size
        print(self.vec_dim, "dimensions")
        return word2vec

    def read_seeds(self, seeds_file: str, holder: Set[Any]) -> None:
        """
        Reads the seeds file and adds the seeds to the holder.
        """
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                ent1 = line.split(";")[0].strip()
                ent2 = line.split(";")[1].strip()
                seed = Seed(ent1, ent2)
                holder.add(seed)
