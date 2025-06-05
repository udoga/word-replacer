from unittest import TestCase
from unittest.mock import Mock
import torch
from torch import Tensor
from source.dropout_substituter import DropoutSubstituter

class DropoutSubstituterTest(TestCase):
    def setUp(self):
        self.model = Mock()
        self.tokenizer = Mock()
        self.substituter = DropoutSubstituter(self.model, self.tokenizer, 0.5, 3)

    def test_applies_dropout(self):
        embedding = torch.ones(768)
        result: Tensor = self.substituter.apply_dropout(embedding, 0.5)
        self.assertEqual(result.shape, embedding.shape)
        self.assertTrue(torch.any(result == 0))
        self.assertTrue(torch.any(result == 1))
