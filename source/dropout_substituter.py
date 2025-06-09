import numpy as np
import pandas as pd
import torch
from torch import Tensor

class DropoutSubstituter:
    def __init__(self, model, dropout_rate, candidate_count):
        self.model = model
        self.dropout_rate = dropout_rate
        self.candidate_count = candidate_count

    def substitute(self, text, target):
        input_ids = self.model.get_encoding_from_text(text)
        target_id = self.model.get_encoding_from_text(target)[1]
        target_index = input_ids.index(target_id)
        original_embeddings = self.model.get_input_embeddings(input_ids)
        original_output = self.model.get_output_from_embeddings(original_embeddings)
        masked_embeddings = self.mask_target_embedding(original_embeddings, target_index, self.dropout_rate)
        masked_output = self.model.get_output_from_embeddings(masked_embeddings)
        prediction_probs = self.get_prediction_probs(masked_output, 0, target_index)
        same_prediction_prob = prediction_probs[target_id].item()
        candidate_ids = torch.topk(prediction_probs, k=self.candidate_count, dim=0).indices
        candidates = self.model.get_tokens_from_ids(candidate_ids.tolist())
        candidate_probs = prediction_probs[candidate_ids].tolist()
        normalized_probs = self.get_normalized_probs(candidate_probs, same_prediction_prob)
        proposal_scores = self.get_proposal_scores(normalized_probs)
        alternative_encodings = self.find_alternative_encodings(input_ids, target_index, candidate_ids)
        alternatives_output = self.model.get_output_from_encodings(alternative_encodings)
        similarity_matrix = self.get_similarity_matrix(original_output, alternatives_output)
        target_similarities = similarity_matrix[:, target_index]
        average_attention_matrix = self.get_average_attention_matrix(original_output)
        validation_scores = torch.matmul(similarity_matrix, average_attention_matrix[target_index])
        return pd.DataFrame(data=dict(
            candidate=candidates,
            candidate_prob=candidate_probs,
            normalized_prob=normalized_probs,
            proposal_score=proposal_scores,
            target_similarity=target_similarities,
            validation_score=validation_scores))

    def mask_target_embedding(self, embeddings, target_index, dropout_rate):
        embedding_copy = embeddings.clone()
        embedding_copy[target_index] = self.apply_dropout(embedding_copy[target_index], dropout_rate)
        return embedding_copy

    def apply_dropout(self, embedding: Tensor, dropout_rate) -> Tensor:
        embedding_length = embedding.shape[0]
        dropout_count = round(dropout_rate * embedding_length)
        dropout_indices = np.random.RandomState(42).choice(embedding_length, dropout_count, replace=False)
        embedding_copy = embedding.clone()
        embedding_copy[dropout_indices] = 0
        return embedding_copy

    def get_proposal_scores(self, normalized_probs) -> np.ndarray:
        return np.log(normalized_probs)

    def get_similarity_matrix(self, original_output, alternatives_output):
        similarity_matrix = []
        cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        token_count = original_output.hidden_states[0].shape[1]
        for token_index in range(token_count):
            original_representation = self.get_contextualized_representations(original_output, token_index)
            alternative_representations = self.get_contextualized_representations(alternatives_output, token_index)
            alternative_similarities_for_token = cos_similarity(original_representation, alternative_representations)
            similarity_matrix.append(alternative_similarities_for_token)
        return torch.stack(similarity_matrix).t()

    def get_prediction_probs(self, output, text_index, target_index) -> Tensor:
        return torch.softmax(output.logits[text_index][target_index], dim=0)

    def get_contextualized_representations(self, output, token_index) -> Tensor:
        return torch.cat(tuple([output.hidden_states[i][:, token_index, :] for i in [-4, -3, -2, -1]]), dim=1)

    def get_normalized_probs(self, prediction_probs, same_prediction_prob) -> np.ndarray:
        return np.array(prediction_probs) / (1.0 - same_prediction_prob)

    def find_alternative_encodings(self, encoding: np.ndarray, target_index, token_ids) -> np.ndarray:
        alternative_encodings = []
        for token_id in token_ids:
            new_encoding = encoding.copy()
            new_encoding[target_index] = token_id
            alternative_encodings.append(new_encoding)
        return np.array(alternative_encodings)

    def get_average_attention_matrix(self, output):
        return torch.div(torch.stack(list(output.attentions)).squeeze().sum(0).sum(0), (12 * 12.0))
