import nltk
import time
import uuid
import logging
import re
import nltk.chunk
import itertools
from nltk.tag import TaggerI, untag
from nltk.chunk import ChunkParserI, tree2conlltags, conlltags2tree
from nltk.stem.wordnet import WordNetLemmatizer
from chunkers import BigramChunker

class Relations():
    chunker = None

    def __init__(self,chunker):
        #self.rpc_client = RpcClient()
        self._logger = logging.getLogger("relations")
        self.chunker = chunker

    def extract_triples(self, sentence, normalize=False, lex_syn_constraints=False, allow_unary=False, short_rel=False):
        np_chunks_tree = self._get_chunks(sentence)
        reverb = Reverb(sentence, np_chunks_tree, normalize, lex_syn_constraints, allow_unary, short_rel)
        triples = reverb.extract_triples()
        return triples

    def _get_chunks(self, sentence):
        l = nltk.pos_tag(nltk.word_tokenize(sentence))
        parsed_sent = self.chunker.parse(l)
        return parsed_sent

class Reverb():
    """
    Approximately Reverb (regex part of it) translation.
    """
    def __init__(self, sentence, np_chunks_tree, normalize, lex_syn_constraints, allow_unary, short_rel):
        """
        :param lex_syn_constraints: Use syntactic and lexical constraints.
        :param allow_unary: allow unary relations ("the book is published").
        """
        self.sentence = sentence
        self.np_chunks_tree = np_chunks_tree
        self.normalize = normalize
        self.lex_syn_constraints = lex_syn_constraints
        self.allow_unary = allow_unary
        self.pos_tags = nltk.pos_tag(nltk.word_tokenize(self.sentence))
        self.short_rel = short_rel
        self._logger = logging.getLogger("relations")

    def extract_triples(self):
        """
        Extracts (subject predicate object) triples from the given sentence. Returns a list of triples.
        """
        # verb = optional adverb [modal or other verbs] optional particle/adverb
        verb = "<RB>?<MD|VB|VBD|VBP|VBZ|VBG|VBN><RP>?<RB>?"
        preposition =  "<RB>?<IN|TO|RP><RB>?"
        word = "<PRP$|CD|DT|JJ|JJS|JJR|NN|NNS|NNP|NNPS|POS|PRP|RB|RBR|RBS|VBN|VBG>"
        cp = None
        if  self.short_rel:
            short_rel_pattern = "(%s(%s)?)+" % (verb, preposition)
            grammar_short = "REL: {%s}" % short_rel_pattern
            cp = nltk.RegexpParser(grammar_short)
        else:
            long_rel_pattern = "(%s(%s*(%s)+)?)+" % (verb, word, preposition)
            grammar_long = "REL: {%s}" % long_rel_pattern
            cp = nltk.RegexpParser(grammar_long)
        tree = cp.parse(self.pos_tags)
        self._logger.info("rel: %s" % tree)
        #triples = self._get_triples(tree)
        triples = self._get_relational_word(tree)
        return triples

    def _get_relational_word(self, tree):
        relation = ""
        left_part = ""
        right_part = ""
        triples = []
        relations = []
        normalized_relations = []
        non_relations = []
        rel_indices = []
        curr = ""
        index = 0
        for t in tree:
            if type(t) == nltk.Tree and t.label() == "REL":
                rel_ind = (index, index + len(t.leaves()))
                rel_indices.append(rel_ind)
                relation = " ".join(map(lambda x : x[0], t.leaves()))
                relations.append(relation)
                if self.normalize:
                    norm_rel = self.verb_rel_normalize(rel_ind)
                else:
                    norm_rel = relation
                normalized_relations.append(norm_rel)
                non_relations.append(curr.strip())
                curr = ""
                index += len(t.leaves())
            else:
                curr += " " + t[0]
                index += 1
        return normalized_relations

    def _get_triples(self, tree):
        relation = ""
        left_part = ""
        right_part = ""
        triples = []
        relations = []
        normalized_relations = []
        non_relations = []
        rel_indices = []
        curr = ""
        index = 0
        for t in tree:
            if type(t) == nltk.Tree and t.label() == "REL":
                rel_ind = (index, index + len(t.leaves()))
                rel_indices.append(rel_ind)
                relation = " ".join(map(lambda x : x[0], t.leaves()))
                relations.append(relation)
                if self.normalize:
                    norm_rel = self.verb_rel_normalize(rel_ind)
                else:
                    norm_rel = relation
                normalized_relations.append(norm_rel)
                non_relations.append(curr.strip())
                curr = ""
                index += len(t.leaves())
            else:
                curr += " " + t[0]
                index += 1
        if curr.strip() != "": # if relation is the last item in the tree drop the following appending
            non_relations.append(curr.strip())
        for ind, rel in enumerate(relations):
            left_part = non_relations[ind]
            right_part = ""
            if ind + 1 < len(non_relations):
                right_part = non_relations[ind + 1]
            else:
                break
            left_arg = self._get_left_arg(rel, rel_indices[ind][0], left_part)
            right_arg = self._get_right_arg(rel, rel_indices[ind][1], right_part)
            if self.lex_syn_constraints:
                if not self._is_relation_lex_syn_valid(rel, rel_indices[ind]):
                    continue
            if not self.allow_unary and right_arg == "":
                continue
            if left_arg == "":
                # todo: try to extract left_arg even the subject is for example before comma like in
                # the following sentence (for "has" relation):
                # This chassis supports up to six fans , has a complete black interior
                continue
            triples.append((left_arg, normalized_relations[ind], right_arg))
        return triples

    def _is_relation_lex_syn_valid(self, relation, rel_indices):
        """
        Checks syntactic and lexical constraints on the relation.
        """
        if len(relation) < 2: # relation shouldn't be a single character
            return False
        pos_tags = map(lambda x : x[1], self.pos_tags[rel_indices[0] : rel_indices[1]])
        rel_words = map(lambda x : x[0], self.pos_tags[rel_indices[0] : rel_indices[1]])
        # these POS tags and words cannot appear in relation: CC, ",", PRP, "that", "if":
        forbidden_tags = ["CC", "PRP"]
        if len(set(pos_tags).intersection(set(forbidden_tags))) > 0:
            return False
        forbidden_words = [",", "that", "if"]
        if len(set(rel_words).intersection(set(forbidden_words))) > 0:
            return False
        # The POS tag of the first verb in the relation cannot be VBG or VBN:
        for tag in pos_tags:
            if tag[:2] == "VB":
                if tag == "VBG" or tag == "VBN":
                    return False
                else:
                    break
        # The previous tag can't be an existential "there" or a TO:
        if self.pos_tags[rel_indices[0] - 1][1] in ["EX", "TO"]:
            return False
        return True

    def _get_left_arg(self, relation, left_rel_border, left_part):
        candidate = ""
        word_ind = 0 #count words to see if we are still on the left side of the relation
        for t in self.np_chunks_tree:
            if type(t) != nltk.Tree:
                word_ind += 1
            else: # don't need to check if t.node == "NP", because we have only an NP chunker here
                word_ind += len(t.leaves())
                #leaves = " ".join(map(lambda x : x.split("/")[0], t.leaves()))
                leaves = " ".join(map(lambda x : x[0], t.leaves()))
                if word_ind > left_rel_border: # check if we are already on the right side of the relation
                    break
                if not leaves in left_part:
                    # could be that leaves are for example part of previous relation or previous left_arg
                    # left_part currently contains only words after previous relation and before current relation
                    # todo: this should be actually changed if we want to be able to
                    # extract subjects that are "far" away from the relation
                    continue
                if len(t) == 1:
                    #word = t[0].split("/")[0]
                    #tag = t[0].split("/")[1]
                    word = t[0][0]
                    tag = t[0][1]
                    if tag == "EX": # first argument can't be an existential "there"
                        continue
                    if tag == "WDT" or tag == "WP$" or tag == "WRB" or tag == "WP": #first argument can't be a Wh word
                        continue
                    if tag == "IN": # first argument can't be a preposition
                        continue
                    if word == "that" or word == "which":
                        continue
                    reflexive_pronouns = ["myself", "yourself", "himself", "herself", "itself", "oneself", "ourselves",\
                                  "ourself", "yourselves", "themselves"]
                    if word in reflexive_pronouns:
                        continue
                cand = leaves
                # First argument can't match "ARG1, REL" "ARG1 and REL" or "ARG1, and REL"
                if re.findall("%s\s*,\s*%s" % (cand, relation), self.sentence):
                    continue
                if re.findall("%s\s*and\s*%s" % (cand, relation), self.sentence):
                    continue
                if re.findall("%s\s*,\s*and\s*%s" % (cand, relation), self.sentence):
                    continue
                arg_indices = (word_ind - len(t.leaves()), word_ind)
                if self.normalize:
                    candidate = self.arg_normalize(arg_indices)
                else:
                    candidate = cand
                # First argument should be closest to relation that passes through filters (means don't break the loop here)
        return candidate

    def _get_right_arg(self, relation, right_rel_border, right_part):
        candidate = ""
        word_ind = 0 #count words to see if we are on the right side of the relation
        for t in self.np_chunks_tree:
            if type(t) != nltk.Tree:
                word_ind += 1
            else:
                word_ind += len(t.leaves())
                #leaves = " ".join(map(lambda x : x.split("/")[0], t.leaves()))
                leaves = " ".join(map(lambda x : x[0], t.leaves()))
                if word_ind < right_rel_border: # check if we are still on the left side of the relation
                    continue
                if not leaves in right_part:
                    # right_part currently contains only words after current relation and before next relation
                    continue
                if len(t) == 1:
                    #word = t[0].split("/")[0]
                    #tag = t[0].split("/")[1]
                    word = t[0][0]
                    tag = t[0][1]
                    if tag == "WDT" or tag == "WP$" or tag == "WRB" or tag == "WP": #first argument can't be a Wh word
                        continue
                    if word == "which":
                        continue
                cand = leaves
                # Second argument should be adjacent to the relation
                if not re.findall("%s\s*%s" % (relation, cand), self.sentence):
                    continue
                arg_indices = (word_ind - len(t.leaves()), word_ind)
                if self.normalize:
                    candidate = self.arg_normalize(arg_indices)
                else:
                    candidate = cand
                # Second argument should be closest to relation that passes through filters (break the loop)
                break
        return candidate

    def verb_rel_normalize(self, rel_indices):
        ignore_pos_tags = []
        ignore_pos_tags.append("MD") # can, must, should
        ignore_pos_tags.append("DT") # the, an, these
        ignore_pos_tags.append("PDT") # predeterminers
        ignore_pos_tags.append("WDT") # wh-determiners
        ignore_pos_tags.append("JJ") # adjectives
        ignore_pos_tags.append("RB") # adverbs
        ignore_pos_tags.append("PRP$") # my, your, our
        # remove leading "be", "have", "do"
        aux_verbs = []
        aux_verbs.append("be");
        aux_verbs.append("have");
        aux_verbs.append("do");
        no_noun = True
        relation = ""
        pos_tags = self.pos_tags[rel_indices[0]:rel_indices[1]]
        for _word, pos_tag in pos_tags:
            if pos_tag[0] == "N":
                no_noun = False
                break
        lmtzr = WordNetLemmatizer()
        for ind, (word, pos_tag) in enumerate(pos_tags):
            is_adj = pos_tag[0] == "J"
            # This is checking for a special case where the relation phrase
            # contains an adjective, but no noun. This covers cases like
            # "is high in" or "looks perfect for" where the adjective carries
            # most of the semantics of the relation phrase. In these cases, we
            # don't want to strip out the adjectives.
            keep_adj = is_adj and no_noun
            if pos_tag in ignore_pos_tags and not keep_adj:
                continue
            else:
                if pos_tag[0] in ["N", "V"]:
                    pos = pos_tag[0].lower()
                new_word = lmtzr.lemmatize(word, pos)
                if new_word in aux_verbs and ind + 1 < len(pos_tags) and pos_tags[ind + 1][1][0] == "V":
                    pass
                else:
                    relation += " " + new_word
        return relation.strip()

    def arg_normalize(self, arg_indices):
        """If the field contains a proper noun, don't normalize
            If the field contains a tag starting with N, return the rightmost one - stemmed
            Otherwise, don't normalize.
         """
        contains_proper_noun = False
        last_noun_index = -1
        start = arg_indices[0]
        end = arg_indices[1]
        pos_tags = self.pos_tags[start:end]
        for ind, (_word, pos_tag) in enumerate(pos_tags):
            if pos_tag in ["NNP", "NNPS"]:
                contains_proper_noun = True
            if pos_tag[0] == "N":
                last_noun_index = ind
        if contains_proper_noun or last_noun_index == -1:
            not_changed = map(lambda x : x[0], self.pos_tags[start:end])
            not_changed = " ".join(not_changed)
            return not_changed
        else:
            last_noun = pos_tags[last_noun_index][0]
            lmtzr = WordNetLemmatizer()
            new_word = lmtzr.lemmatize(last_noun, "n")
            return new_word
