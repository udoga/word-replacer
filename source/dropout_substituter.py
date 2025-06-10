import numpy as np
import torch
from torch import Tensor

from substitution_table import SubstitutionTable

class DropoutSubstituter:
    def __init__(self, model, dropout_rate, candidate_count):
        self.model = model
        self.dropout_rate = dropout_rate
        self.candidate_count = candidate_count

    def substitute(self, text, target):
        t = SubstitutionTable()
        token_ids = self.model.get_encoding_from_text(text)
        target_id = self.model.get_encoding_from_text(target)[1]
        target_index = token_ids.index(target_id)
        clear_embeddings = self.model.get_input_embeddings(token_ids)
        clear_output = self.model.get_output_from_embeddings(clear_embeddings)
        masked_embeddings = self.mask_target_embedding(clear_embeddings, target_index, self.dropout_rate)
        masked_output = self.model.get_output_from_embeddings(masked_embeddings)
        prediction_probs = self.get_prediction_probs(masked_output, 0, target_index)
        candidate_ids = torch.topk(prediction_probs, k=self.candidate_count, dim=0).indices
        t.candidate_tokens = self.model.get_tokens_from_ids(candidate_ids.tolist())
        t.candidate_probs = prediction_probs[candidate_ids]
        t.normalized_probs = self.get_normalized_probs(t.candidate_probs, prediction_probs[target_id].item())
        t.proposal_scores = torch.log(t.normalized_probs)
        alternative_encodings = self.find_alternative_encodings(token_ids, target_index, candidate_ids)
        alternative_output = self.model.get_output_from_encodings(alternative_encodings)
        alternative_tokens_similarities = self.get_alternative_tokens_similarities(clear_output, alternative_output)
        t.target_similarities = alternative_tokens_similarities[:, target_index]
        token_target_attentions = self.get_average_attention_matrix(clear_output)[target_index]
        t.validation_scores = torch.matmul(alternative_tokens_similarities, token_target_attentions)
        return t

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

    def get_alternative_tokens_similarities(self, original_output, alternatives_output) -> Tensor:
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

    def get_normalized_probs(self, candidate_probs: Tensor, original_prediction_prob) -> Tensor:
        return torch.div(candidate_probs, (1.0 - original_prediction_prob))

    def find_alternative_encodings(self, encoding, target_index, token_ids) -> Tensor:
        alternative_encodings = []
        for token_id in token_ids:
            new_encoding = encoding.copy()
            new_encoding[target_index] = token_id
            alternative_encodings.append(new_encoding)
        return torch.tensor(alternative_encodings)

    def get_average_attention_matrix(self, output) -> Tensor:
        return torch.div(torch.stack(list(output.attentions)).squeeze().sum(0).sum(0), (12 * 12.0))
