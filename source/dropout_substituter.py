import torch
from torch import Tensor

from substitution_table import SubstitutionTable

class DropoutSubstituter:
    def __init__(self, model, dropout_rate = 0.3, candidate_count = 10, alpha = 0.01, iteration_count=1,
                 deterministic=True):
        self.model = model
        self.dropout_rate = dropout_rate
        self.candidate_count = candidate_count
        self.alpha = alpha
        self.iteration_count = iteration_count
        self.deterministic = deterministic

    def substitute(self, text, target) -> SubstitutionTable:
        substitution_tables = []
        for i in range(self.iteration_count):
            substitution_tables.append(self.substitute_once(text, target, i))
        return SubstitutionTable.avg_tables(substitution_tables, 'final_score', self.candidate_count)

    def substitute_once(self, text, target, iteration_index) -> SubstitutionTable:
        t = SubstitutionTable()
        token_ids = self.model.get_encoding_from_text(text)
        target_id = self.model.get_encoding_from_text(target)[1]
        target_index = token_ids.index(target_id)
        clear_embeddings = self.model.get_input_embeddings(token_ids)
        clear_output = self.model.get_output_from_embeddings(clear_embeddings)
        masked_embeddings = self.mask_target(clear_embeddings, target_index, self.dropout_rate, iteration_index)
        masked_output = self.model.get_output_from_embeddings(masked_embeddings)
        prediction_probs = self.get_prediction_probs(masked_output, 0, target_index)
        candidate_ids = torch.topk(prediction_probs, k=self.candidate_count, dim=0).indices
        t['candidate'] = self.model.get_tokens_from_ids(candidate_ids.tolist())
        t['candidate_prob'] = prediction_probs[candidate_ids]
        t['normalized_prob'] = self.get_normalized_probs(t['candidate_prob'], prediction_probs[target_id].item())
        t['proposal_score'] = torch.log(t['normalized_prob'])
        alternative_encodings = self.find_alternative_encodings(token_ids, target_index, candidate_ids)
        alternative_output = self.model.get_output_from_encodings(alternative_encodings)
        alternative_tokens_similarities = self.get_alternative_tokens_similarities(clear_output, alternative_output)
        t['target_similarity'] = alternative_tokens_similarities[:, target_index]
        token_target_attentions = self.get_average_attention_matrix(clear_output)[target_index]
        t['validation_score'] = torch.matmul(alternative_tokens_similarities, token_target_attentions)
        t['final_score'] = t['validation_score'] + self.alpha * t['proposal_score']
        return t

    def mask_target(self, embeddings, target_index, dropout_rate, iteration_index):
        embeddings_copy = embeddings.clone()
        self.apply_dropout(embeddings_copy[target_index], dropout_rate, iteration_index)
        return embeddings_copy

    def apply_dropout(self, embedding: Tensor, dropout_rate, iteration_index):
        embedding_length = embedding.shape[0]
        dropout_count = round(dropout_rate * embedding_length)
        generator = torch.Generator().manual_seed(iteration_index) if self.deterministic else None
        dropout_indices = torch.randperm(embedding_length, generator=generator)[:dropout_count]
        embedding[dropout_indices] = 0

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
