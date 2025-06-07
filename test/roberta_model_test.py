from torch import Tensor
from unittest import TestCase
from source.roberta_model import RobertaModel

class RobertaModelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = RobertaModel()

    def test_gets_token_ids_from_text(self):
        self.assertEqual([0, 20760, 2], self.model.get_encoding_from_text("hello"))
        self.assertEqual([0, 20760, 232, 2], self.model.get_encoding_from_text("hello world"))

    def test_gets_tokens_from_ids(self):
        self.assertEqual(["hello"], self.model.get_tokens_from_ids([20760]))
        self.assertEqual(["hello", "world"], self.model.get_tokens_from_ids([20760, 232]))

    def test_gets_input_embeddings_from_encoding(self):
        embeddings: Tensor = self.model.get_input_embeddings([0, 20760, 2])
        batch_embeddings: Tensor = self.model.get_batch_input_embeddings([[0, 20760, 2], [0, 232, 2]])
        self.assertEqual((3, 768), embeddings.shape)
        self.assertEqual((2, 3, 768), batch_embeddings.shape)
