import numpy as np
from numpy.testing import assert_array_equal
import torch
from torch import Tensor
from unittest import TestCase
from source.dropout_substituter import DropoutSubstituter
from source.roberta_model import RobertaModel

class DropoutSubstituterTest(TestCase):
    def setUp(self):
        self.model = RobertaModel
        self.substituter = DropoutSubstituter(self.model, 0.5, 3)

    def test_applies_dropout(self):
        embedding = torch.ones(768)
        result: Tensor = self.substituter.apply_dropout(embedding, 0.5)
        self.assertEqual(result.shape, embedding.shape)
        self.assertTrue(torch.any(result == 0))
        self.assertTrue(torch.any(result == 1))

    def test_finds_alternative_encodings(self):
        encoding = np.array([1, 2, 3])
        alternative_encodings = self.substituter.find_alternative_encodings(encoding, 0, [100, 101])
        assert_array_equal(alternative_encodings, np.array([[100, 2, 3], [101, 2, 3]]))
