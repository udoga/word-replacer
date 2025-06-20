from unittest import TestCase
from source.lemmatizer import Lemmatizer

class LemmatizerTest(TestCase):
    def setUp(self):
        self.lemmatizer = Lemmatizer()

    def test_finds_if_words_has_same_root(self):
        self.assertTrue(self.lemmatizer.has_same_root("film", "films"))
        self.assertTrue(self.lemmatizer.has_same_root("tell", "told"))
        self.assertFalse(self.lemmatizer.has_same_root("tell", "film"))
