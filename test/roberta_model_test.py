import torch
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

    def test_gets_input_embeddings_from_ids(self):
        embeddings = self.model.get_input_embeddings([0, 20760, 2])
        self.assertTrue(torch.is_tensor(embeddings))
        self.assertEqual((3, 768), embeddings.shape)

    def test_gets_predictions_logits(self):
        embeddings = self.model.get_input_embeddings([0, 20760, 2])
        logits = self.model.get_prediction_logits(embeddings, 1)
        self.assertTrue(torch.is_tensor(logits))
        self.assertEqual((self.model.get_vocabulary_size(),), logits.shape)
