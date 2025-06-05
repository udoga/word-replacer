import numpy as np
import torch
from torch import Tensor

class DropoutSubstituter:
    def __init__(self, model, tokenizer, dropout_rate, candidate_count):
        self.model = model
        self.tokenizer = tokenizer
        self.dropout_rate = dropout_rate
        self.candidate_count = candidate_count

    def substitute(self, text, target):
        input_ids = self.get_token_ids(text)
        target_id = self.get_token_ids(target)[1]
        target_position = input_ids.index(target_id)
        input_embeddings = self.get_input_embeddings(input_ids)
        self.apply_dropout(input_embeddings[target_position], self.dropout_rate)
        prediction_probs = self.get_prediction_probs(input_embeddings, target_position)
        same_prediction_prob = prediction_probs[target_id].item()
        candidate_ids = torch.topk(prediction_probs, k=self.candidate_count, dim=0).indices
        candidates = [c.strip() for c in self.tokenizer.batch_decode(candidate_ids)]
        candidate_probs = prediction_probs[candidate_ids].tolist()
        candidate_proposal_scores = self.get_proposal_scores(candidate_probs, same_prediction_prob)
        return dict(zip(candidates, candidate_proposal_scores.tolist()))

    def get_token_ids(self, text):
        return self.tokenizer.encode(" " + text)

    def get_input_embeddings(self, token_ids) -> Tensor:
        with torch.no_grad():
            return self.model.get_input_embeddings()(torch.tensor([token_ids]))[0]

    def get_prediction_probs(self, input_embeddings: Tensor, target_position) -> Tensor:
        with torch.no_grad():
            model_output = self.model(inputs_embeds=input_embeddings.unsqueeze(0))
        logits_per_output = model_output.logits[0]
        prediction_logits = logits_per_output[target_position]
        return torch.softmax(prediction_logits, dim=0)

    def apply_dropout(self, embedding: Tensor, dropout_rate):
        embedding_length = embedding.shape[0]
        dropout_count = round(dropout_rate * embedding_length)
        dropout_indices = np.random.RandomState(0).choice(embedding_length, dropout_count, replace=False)
        embedding_copy = embedding.clone()
        embedding_copy[dropout_indices] = 0
        return embedding_copy

    def get_proposal_scores(self, prediction_probs, same_prediction_prob):
        return np.log(self.get_normalized_probs(prediction_probs, same_prediction_prob))

    def get_normalized_probs(self, prediction_probs, same_prediction_prob):
        return np.array(prediction_probs) / (1.0 - same_prediction_prob)
