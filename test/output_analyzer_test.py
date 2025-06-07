import torch
from torch import Tensor, testing
from unittest import TestCase

from output_analyzer import OutputAnalyzer
from source.roberta_model import RobertaModel

class RobertaModelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = RobertaModel()
        cls.output = cls.model.get_output_from_encodings([[0, 20760, 2], [0, 232, 2]])
        cls.analyzer = OutputAnalyzer(cls.output)

    def test_finds_prediction_logits(self):
        logits: Tensor = self.analyzer.get_prediction_logits(0, 1)
        self.assertEqual((self.model.get_vocabulary_size(),), logits.shape)

    def test_finds_contextual_representations(self):
        states = self.output.hidden_states
        representations: Tensor = self.analyzer.get_contextualized_representations(1, 2)
        testing.assert_close(torch.concat([states[-1][0][1], states[-2][0][1]], dim=0), representations[0])
        testing.assert_close(torch.concat([states[-1][1][1], states[-2][1][1]], dim=0), representations[1])
