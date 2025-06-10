import numpy as np
from numpy.testing import assert_array_equal
import torch
from torch import Tensor
from unittest import TestCase
from source.dropout_substituter import DropoutSubstituter
from source.roberta_model import RobertaModel

class DropoutSubstituterTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = RobertaModel()

    def setUp(self):
        self.substituter = DropoutSubstituter(self.model, 0.3, 3, 0.003)

    def test_applies_dropout(self):
        embedding = torch.ones(768)
        self.substituter.apply_dropout(embedding, 0.5)
        self.assertTrue(torch.any(embedding == 0))
        self.assertTrue(torch.any(embedding == 1))

    def test_finds_alternative_encodings(self):
        encoding = np.array([1, 2, 3])
        alternative_encodings = self.substituter.find_alternative_encodings(encoding, 0, [100, 101])
        assert_array_equal(alternative_encodings, np.array([[100, 2, 3], [101, 2, 3]]))

    def test_finds_prediction_logits(self):
        output = self.model.get_output_from_encodings(torch.tensor([[0, 20760, 2], [0, 232, 2]]))
        probs: Tensor = self.substituter.get_prediction_probs(output, 0, 1)
        self.assertEqual((self.model.get_vocabulary_size(),), probs.shape)
