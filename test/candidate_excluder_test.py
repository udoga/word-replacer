from source.candidate_excluder import CandidateExcluder
from unittest import TestCase

class CandidateExcluderTest(TestCase):
    def setUp(self):
        self.excluder = CandidateExcluder()

    def test_excludes_continuation_token_candidates(self):
        self.assertEqual([False], self.excluder.are_candidates_included(["nery"]))

    def test_excludes_candidates_that_has_same_root_with_target(self):
        self.assertEqual([False, True], self.excluder.are_candidates_included([" stronger", " powerful"], "strong"))

    def test_excludes_punctuation_candidates(self):
        self.assertEqual([True, False, False], self.excluder.are_candidates_included([" powerful", " .", " .."]))

    def test_excludes_stopword_candidates(self):
        self.assertEqual([False, True, False], self.excluder.are_candidates_included([" or", " powerful", " And"]))

    def test_excludes_candidates_with_different_pos_tag(self):
        self.assertEqual([False], self.excluder.are_candidates_included([" powering"], "powerful", "a"))

    def test_finds_if_words_has_same_root(self):
        self.assertTrue(self.excluder.has_same_root("film", "films"))
        self.assertTrue(self.excluder.has_same_root("tell", "told"))
        self.assertFalse(self.excluder.has_same_root("tell", "film"))
