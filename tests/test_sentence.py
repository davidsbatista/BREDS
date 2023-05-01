from breds.sentence import Entity


def test_entity():
    ent1 = Entity("Paris", ["Paris"], "Location", [2, 4])
    ent2 = Entity("France", ["France"], "Country", [2, 4])
    ent3 = Entity("Paris", ["Paris"], "Location", [2, 4])

    assert ent1 == ent3


# class TestRelationship(unittest.TestCase):
#     def setUp(self):
#         self.before = [("the", "DT"), ("capital", "NN"), ("city", "NN")]
#         self.between = [("of", "IN")]
#         self.after = [("is", "VBZ"), ("a", "DT"), ("beautiful", "JJ"), ("place", "NN")]
#         self.rel1 = Relationship(
#             "Paris is the capital city of France. It is a beautiful place",
#             self.before,
#             self.between,
#             self.after,
#             "Paris",
#             "France",
#             "Location",
#             "Country",
#         )
#         self.rel2 = Relationship(
#             "Paris is the capital city of France. It is a beautiful place",
#             self.before,
#             self.between,
#             self.after,
#             "Paris",
#             "Italy",
#             "Location",
#             "Country",
#         )
#         self.rel3 = Relationship(
#             "Paris is the capital city of Italy. It is a beautiful place",
#             self.before,
#             self.between,
#             self.after,
#             "Paris",
#             "France",
#             "Location",
#             "Country",
#         )
#
#     def test_equality(self):
#         self.assertEqual(self.rel1, self.rel1)
#         self.assertNotEqual(self.rel1, self.rel2)
#         self.assertNotEqual(self.rel1, self.rel3)
#
#     def test_hash(self):
#         self.assertEqual(hash(self.rel1), hash(self.rel1))
#         self.assertNotEqual(hash(self.rel1), hash(self.rel2))
#         self.assertNotEqual(hash(self.rel1), hash(self.rel3))


# class TestSentence(unittest.TestCase):
#     def setUp(self):
#         self.max_tokens = 5
#         self.min_tokens = 2
#         self.window_size = 1
#         self.pos_tagger = None
#         self.sentence_text = "Paris is the capital city of France. It is a beautiful place"
#         self.sentence = Sentence(
#             self.sentence_text,
#             "Location",
#             "Country",
#             self.max_tokens,
#             self.min_tokens,
#             self.window_size,
#             self.pos_tagger,
#         )
#
#     def test_relationship_extraction(self):
#         expected_relationships = [
#             Relationship(
#                 "Paris is the capital city of France. It is a beautiful place",
#                 [("the", "DT"), ("capital", "NN"), ("city", "NN")],
#                 [("of", "IN")],
#                 [("is", "VBZ"), ("a", "DT"), ("beautiful", "JJ"), ("place", "NN")],
#                 "Paris",
#                 "France",
#                 "Location",
#                 "Country",
#             )
#         ]
#         self.assertListEqual(self.sentence.relationships, expected_relationships)
#
#     def test_find_locations(self):
#         text_tokens = [
#             "Paris",
#             "is",
#             "the",
#             "capital",
#             "city",
#             "of",
#             "France",
#             ".",
#             "It",
#             "is",
#             "a",
#             "beautiful",
#             "place",
#         ]
#         expected_locations = (["Paris"], [0])
#         self.assertEqual(find_locations("Paris", text_tokens), expected_locations)
