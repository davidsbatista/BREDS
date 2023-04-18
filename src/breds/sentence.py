import re

from nltk import word_tokenize
from nltk.corpus import stopwords

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

# tokens between entities which do not represent relationships
bad_tokens = [",", "(", ")", ";", "''", "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords = stopwords.words("english")
not_valid = bad_tokens + stopwords
regex_clean_tags = re.compile("</?[A-Z]+>", re.U)


def tokenize_entity(entity):
    """Simple tokenize an entity string"""
    parts = word_tokenize(entity)
    if parts[-1] == ".":
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string, text_tokens):
    """Find the locations of an entity in a text."""
    locations = []
    ent_parts = tokenize_entity(entity_string)
    for idx in range(len(text_tokens)):
        if text_tokens[idx : idx + len(ent_parts)] == ent_parts:
            locations.append(idx)
    return ent_parts, locations


class Entity:
    """Entity class to hold information about an entity extracted from a sentence."""

    def __init__(self, surface_string, surface_string_parts, ent_type, locations) -> None:
        self.string = surface_string
        self.parts = surface_string_parts
        self.type = ent_type
        self.locations = locations

    def __hash__(self) -> int:
        return hash(self.string) ^ hash(self.type)

    def __eq__(self, other) -> bool:
        return self.string == other.string and self.type == other.type


class Relationship:  # pylint: disable=too-many-arguments, too-many-instance-attributes
    """Relationship class to hold information about a relationship extracted from a sentence."""

    def __init__(self, sentence, before, between, after, ent1, ent2, e1_type, e2_type):
        self.sentence = sentence
        self.before = before
        self.between = between
        self.after = after
        self.ent1 = ent1
        self.ent2 = ent2
        self.e1_type = e1_type
        self.e2_type = e2_type

    def __eq__(self, other) -> bool:
        return (
            self.ent1 == other.ent1
            and self.before == other.before
            and self.between == other.between
            and self.after == other.after
        )

    def __hash__(self) -> int:
        return hash(self.ent1) ^ hash(self.ent2) ^ hash(self.before) ^ hash(self.between) ^ hash(self.after)


class Sentence:  # pylint: disable=too-few-public-methods, too-many-locals, too-many-arguments
    """Sentence class to hold information about a sentence extracted from a document."""

    def __init__(self, sentence, e1_type, e2_type, max_tokens, min_tokens, window_size, pos_tagger=None):  # noqa: C901
        self.relationships = []
        self.tagged_text = None
        self.entities_regex = re.compile("<[A-Z]+>[^<]+</[A-Z]+>", re.U)
        entities = list(re.finditer(self.entities_regex, sentence))

        if len(entities) >= 2:
            sentence_no_tags = re.sub(regex_clean_tags, "", sentence)  # clean tags from text
            text_tokens = word_tokenize(sentence_no_tags)

            # extract information about the entity, create an Entity instance
            # and store in a structure to hold information collected about
            # all the entities in the sentence
            entities_info = set()
            for ent in entities:
                entity = ent.group()
                e_string = re.findall("<[A-Z]+>([^<]+)</[A-Z]+>", entity)[0]
                e_type = re.findall("<([A-Z]+)", entity)[0]
                e_parts, locations = find_locations(e_string, text_tokens)
                ent = Entity(e_string, e_parts, e_type, locations)
                entities_info.add(ent)

            # create a hash table:
            # - key is the starting index in the tokenized sentence of an entity
            # - value the corresponding Entity instance
            locations = {}
            for ent in entities_info:
                for start in ent.locations:
                    locations[start] = ent

            # look for a pair of entities such that:
            # the distance between the two entities is less than 'max_tokens'
            # and greater than 'min_tokens'
            # the arguments match the seeds semantic types
            sorted_keys = list(sorted(locations))

            for i in range(len(sorted_keys) - 1):
                distance = sorted_keys[i + 1] - sorted_keys[i]
                ent1 = locations[sorted_keys[i]]
                ent2 = locations[sorted_keys[i + 1]]

                if max_tokens >= distance >= min_tokens and ent1.type == e1_type and ent2.type == e2_type:
                    # ignore relationships between the same entity
                    if ent1.string == ent2.string:
                        continue

                    # run PoS-tagger over the sentence only once
                    if self.tagged_text is None:
                        # split text into tokens and tag them using NLTK's default English tagger
                        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/
                        # english.pickle'
                        self.tagged_text = pos_tagger.tag(text_tokens)

                    before = self.tagged_text[: sorted_keys[i]]
                    before = before[-window_size:]
                    between = self.tagged_text[sorted_keys[i] + len(ent1.parts) : sorted_keys[i + 1]]
                    after = self.tagged_text[sorted_keys[i + 1] + len(ent2.parts) :]
                    after = after[:window_size]

                    # ignore relationships where BET context is only stopwords or other invalid words
                    if all(x in not_valid for x in text_tokens[sorted_keys[i] + len(ent1.parts) : sorted_keys[i + 1]]):
                        continue

                    rel = Relationship(sentence, before, between, after, ent1.string, ent2.string, e1_type, ent2.type)
                    self.relationships.append(rel)
