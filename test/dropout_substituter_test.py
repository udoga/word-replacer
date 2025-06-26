import numpy as np
from numpy.testing import assert_array_equal
import torch
from torch import Tensor
from unittest import TestCase
from transformers import RobertaTokenizer, RobertaForMaskedLM

from source.dropout_substituter import DropoutSubstituter

class DropoutSubstituterTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
        cls.model = RobertaForMaskedLM.from_pretrained('roberta-base',
                                                        output_hidden_states=True,
                                                        output_attentions=True,
                                                        attn_implementation="eager")

    def setUp(self):
        self.substituter = DropoutSubstituter(self.tokenizer, self.model)

    def test_gets_token_ids_from_text(self):
        self.assertEqual([0, 20760, 2], self.substituter.get_token_ids_from_text("hello"))
        self.assertEqual([0, 20760, 232, 2], self.substituter.get_token_ids_from_text("hello world"))
        self.assertEqual("finally", self.substituter.get_tokens_from_text("finally")[1])

    def test_gets_tokens_from_ids(self):
        self.assertEqual(["hello"], self.substituter.get_tokens_from_ids([20760]))
        self.assertEqual(["hello", "world"], self.substituter.get_tokens_from_ids([20760, 232]))

    def test_gets_input_embeddings_from_encoding(self):
        embeddings: Tensor = self.substituter.get_input_embeddings([0, 20760, 2])
        batch_embeddings: Tensor = self.substituter.get_batch_input_embeddings([[0, 20760, 2], [0, 232, 2]])
        self.assertEqual((3, 768), embeddings.shape)
        self.assertEqual((2, 3, 768), batch_embeddings.shape)

    def test_applies_dropout(self):
        embedding = torch.ones(768)
        self.substituter.apply_dropout(embedding, 0.5, 0)
        self.assertTrue(torch.any(embedding == 0))
        self.assertTrue(torch.any(embedding == 1))

    def test_finds_alternative_encodings(self):
        encoding = np.array([1, 2, 3])
        alternative_encodings = self.substituter.find_alternative_encodings(encoding, 0, [100, 101])
        assert_array_equal(alternative_encodings, np.array([[100, 2, 3], [101, 2, 3]]))

    def test_find_probabilities_for_each_token_in_vocabulary(self):
        output = self.substituter.get_output_from_encodings(torch.tensor([[0, 42891, 2], [0, 232, 2]]))
        probs: Tensor = self.substituter.get_vocab_probs(output, 0, 1)
        self.assertEqual((self.substituter.get_vocabulary_size(),), probs.shape)

    def test_finds_token_index(self):
        self.assertEqual(1, self.substituter.find_token_index("hello", 0))
        self.assertEqual(3, self.substituter.find_token_index("he was bright and independent", 2))
        self.assertEqual(6, self.substituter.find_token_index("film literature cyberplace includes film reviews", 4))

    def test_compares_given_target_and_found_target(self):
        text = "so , unlike studio films , independent films"
        self.assertRaises(AssertionError, self.substituter.substitute, text, "film", 0)
        self.assertRaises(AssertionError, self.substituter.substitute, text, "film", 3)
        self.substituter.substitute(text, "film", 4) # no error
        self.substituter.substitute(text, "film", 7) # no error

    def test_finds_if_words_has_same_root(self):
        self.assertTrue(self.substituter.has_same_root("film", "films"))
        self.assertTrue(self.substituter.has_same_root("tell", "told"))
        self.assertFalse(self.substituter.has_same_root("tell", "film"))

    def test_removes_candidates_with_same_root_with_target(self):
        self.assertEqual(["end", "edge"], self.substituter.filter_candidates(["end", "Side", "sides", "Edge"], "side"))

    def test_removes_punctuation_candidates(self):
        self.assertEqual(["film"], self.substituter.filter_candidates([".", "..", "film"], "movie"))

    def test_duplicates_sentence_when_concatenation_is_enabled(self):
        substituter = DropoutSubstituter(self.tokenizer, self.model, concatenate=True)
        self.assertEqual([0, 20760, 232, 2, 2, 20760, 232, 2], substituter.get_token_ids_from_text("hello world"))
        self.assertEqual(4, substituter.find_token_index("hello", 0))
