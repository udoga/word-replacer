import numpy as np
import torch
from torch import Tensor

class DropoutSubstituter:
    def __init__(self, model, dropout_rate, candidate_count):
        self.model = model
        self.dropout_rate = dropout_rate
        self.candidate_count = candidate_count

    def substitute(self, text, target):
        input_ids = self.model.get_token_ids_from_text(text)
        target_id = self.model.get_token_ids_from_text(target)[1]
        target_position = input_ids.index(target_id)
        input_embeddings = self.model.get_input_embeddings(input_ids)
        input_embeddings[target_position] = self.apply_dropout(input_embeddings[target_position], self.dropout_rate)
        prediction_logits = self.model.get_prediction_logits(input_embeddings, target_position)
        prediction_probs = torch.softmax(prediction_logits, dim=0)
        same_prediction_prob = prediction_probs[target_id].item()
        candidate_ids = torch.topk(prediction_probs, k=self.candidate_count, dim=0).indices
        candidates = self.model.get_tokens_from_ids(candidate_ids.tolist())
        candidate_probs = prediction_probs[candidate_ids].tolist()
        candidate_proposal_scores = self.get_proposal_scores(candidate_probs, same_prediction_prob)
        return dict(zip(candidates, candidate_probs))

    def apply_dropout(self, embedding: Tensor, dropout_rate) -> Tensor:
        embedding_length = embedding.shape[0]
        dropout_count = round(dropout_rate * embedding_length)
        dropout_indices = np.random.choice(embedding_length, dropout_count, replace=False)
        embedding_copy = embedding.clone()
        embedding_copy[dropout_indices] = 0
        return embedding_copy

    def get_proposal_scores(self, prediction_probs, same_prediction_prob) -> np.ndarray:
        return np.log(self.get_normalized_probs(prediction_probs, same_prediction_prob))

    def get_normalized_probs(self, prediction_probs, same_prediction_prob) -> np.ndarray:
        return np.array(prediction_probs) / (1.0 - same_prediction_prob)

    def find_alternative_encodings(self, encoding: np.ndarray, position, token_ids) -> np.ndarray:
        alternative_encodings = []
        for token_id in token_ids:
            new_encoding = encoding.copy()
            new_encoding[position] = token_id
            alternative_encodings.append(new_encoding)
        return np.array(alternative_encodings)
