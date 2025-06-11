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
        cls.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        cls.model = RobertaForMaskedLM.from_pretrained('roberta-base',
                                                        output_hidden_states=True,
                                                        output_attentions=True,
                                                        attn_implementation="eager")

    def setUp(self):
        self.substituter = DropoutSubstituter(self.tokenizer, self.model)

    def test_gets_token_ids_from_text(self):
        self.assertEqual([0, 20760, 2], self.substituter.get_encoding_from_text("hello"))
        self.assertEqual([0, 20760, 232, 2], self.substituter.get_encoding_from_text("hello world"))

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

    def test_finds_prediction_logits(self):
        output = self.substituter.get_output_from_encodings(torch.tensor([[0, 20760, 2], [0, 232, 2]]))
        probs: Tensor = self.substituter.get_prediction_probs(output, 0, 1)
        self.assertEqual((self.substituter.get_vocabulary_size(),), probs.shape)
