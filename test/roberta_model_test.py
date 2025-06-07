import torch
from torch import Tensor, testing
from unittest import TestCase
from source.roberta_model import RobertaModel

class RobertaModelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = RobertaModel()

    def test_gets_token_ids_from_text(self):
        self.assertEqual([0, 20760, 2], self.model.get_token_ids_from_text("hello"))
        self.assertEqual([0, 20760, 232, 2], self.model.get_token_ids_from_text("hello world"))

    def test_gets_tokens_from_ids(self):
        self.assertEqual(["hello"], self.model.get_tokens_from_ids([20760]))
        self.assertEqual(["hello", "world"], self.model.get_tokens_from_ids([20760, 232]))

    def test_gets_input_embeddings_from_encoding(self):
        embeddings: Tensor = self.model.get_input_embeddings([0, 20760, 2])
        batch_embeddings: Tensor = self.model.get_batch_input_embeddings([[0, 20760, 2], [0, 232, 2]])
        self.assertEqual((3, 768), embeddings.shape)
        self.assertEqual((2, 3, 768), batch_embeddings.shape)

    def test_gets_predictions_logits(self):
        embeddings: Tensor = self.model.get_input_embeddings([0, 20760, 2])
        logits: Tensor = self.model.get_prediction_logits(embeddings, 1)
        self.assertEqual((self.model.get_vocabulary_size(),), logits.shape)

    def test_loads_encodings(self):
        self.model.load_encodings([[0, 20760, 2], [0, 232, 2]])
        hidden_states: tuple = self.model.get_hidden_states()
        self.assertEqual(13, len(hidden_states))
        self.assertEqual((2, 3, 768), hidden_states[0].shape)

    def test_finds_contextual_representation(self):
        encodings = [[0, 20760, 2], [0, 232, 2]]
        self.model.load_encodings(encodings)
        states: tuple = self.model.get_hidden_states()
        representations: Tensor = self.model.get_contextualized_representations(1, 2)
        testing.assert_close(torch.concat([states[-1][0][1], states[-2][0][1]], dim=0), representations[0])
        testing.assert_close(torch.concat([states[-1][1][1], states[-2][1][1]], dim=0), representations[1])
