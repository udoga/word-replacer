import torch
from torch import Tensor, Generator
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.stem import WordNetLemmatizer
from transformers import RobertaForMaskedLM, BertForMaskedLM
from source.candidate_excluder import CandidateExcluder
from source.substitution_table import SubstitutionTable
from source.substitution_request import SubstitutionRequest

class BertSubstituter:
    def __init__(self, model_name, dropout_rate = 0.3, candidate_count = 50, alpha = 0.01, iteration_count=1,
                 deterministic=True, concatenate=False, use_mask_token=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, add_prefix_space=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True,
                attn_implementation="eager").to(torch.get_default_device())
        self.lemmatizer = WordNetLemmatizer()
        self.excluder = CandidateExcluder()
        self.dropout_rate = dropout_rate
        self.candidate_count = candidate_count
        self.alpha = alpha
        self.iteration_count = iteration_count
        self.deterministic = deterministic
        self.concatenate = concatenate
        self.use_mask_token = use_mask_token
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mask_embedding = self.get_input_embedding(self.tokenizer.mask_token_id)

    def substitute(self, r: SubstitutionRequest) -> SubstitutionTable:
        t = SubstitutionTable()
        token_ids = self.get_token_ids_from_text(r.text)
        tokens = self.get_tokens_from_ids(token_ids)
        target_index = self.find_token_index(r.text, r.position)
        target_id = token_ids[target_index]
        target_token = tokens[target_index].strip()
        assert self.excluder.has_same_root(r.target, target_token), f"Target {r.target} != {target_token} at {target_index}"
        clear_embeddings = self.get_input_embeddings(token_ids)
        clear_output = self.get_output_from_embeddings(clear_embeddings)
        vocab_probs = self.get_average_vocab_probs(clear_embeddings, target_index)
        candidate_ids = torch.topk(vocab_probs, k=self.candidate_count, dim=0).indices
        candidate_tokens = self.get_tokens_from_ids(candidate_ids.tolist())
        t['candidate'] = [t.strip() for t in candidate_tokens]
        t['is_included'] = self.excluder.are_candidates_included(candidate_tokens, r.target, r.tag)
        t['candidate_prob'] = vocab_probs[candidate_ids].cpu()
        t['normalized_prob'] = self.get_normalized_probs(t['candidate_prob'], vocab_probs[target_id].item())
        t['proposal_score'] = torch.log(t['normalized_prob'])
        alternative_encodings = self.find_alternative_encodings(token_ids, target_index, candidate_ids)
        alternatives_output = self.get_output_from_encodings(alternative_encodings)
        alternatives_token_similarities = self.get_alternatives_token_similarities(clear_output, alternatives_output)
        t['cls_similarity'] = alternatives_token_similarities[:, 0].cpu()
        t['target_similarity'] = alternatives_token_similarities[:, target_index].cpu()
        token_target_attentions = self.get_average_attention_matrix(clear_output)[:, target_index]
        token_target_weights = token_target_attentions / token_target_attentions.sum()
        t['validation_score'] = torch.matmul(alternatives_token_similarities, token_target_weights).cpu()
        t['final_score'] = t['target_similarity'] + self.alpha * t['proposal_score'].cpu()
        return SubstitutionTable.from_frame(t.to_frame().sort_values(by=["final_score"], ascending=False))

    def get_output_from_encodings(self, encodings):
        with torch.no_grad():
            return self.model(encodings)

    def get_output_from_embeddings(self, embeddings):
        with torch.no_grad():
            return self.model(inputs_embeds=embeddings.unsqueeze(0))

    def get_vocabulary_size(self):
        return len(self.tokenizer)

    def find_token_index(self, text, position):
        words = text.split()
        text_before = " ".join(words[:position+1])
        encoding = self.tokenizer.encode(text, text_before) if self.concatenate else self.tokenizer.encode(text_before)
        target_encoding_length = len(self.tokenizer.encode(words[position])) - 2
        return len(encoding) - 2 - (target_encoding_length - 1)

    def get_tokens_from_text(self, text):
        return self.get_tokens_from_ids(self.get_token_ids_from_text(text))

    def get_token_ids_from_text(self, text):
        return self.tokenizer.encode(text, text) if self.concatenate else self.tokenizer.encode(text)

    def get_tokens_from_ids(self, token_ids):
        return [self.normalize_token(t) for t in self.tokenizer.convert_ids_to_tokens(token_ids)]

    def normalize_token(self, token):
        if isinstance(self.model, RobertaForMaskedLM): return token.replace("Ġ", " ")
        if isinstance(self.model, BertForMaskedLM): return (" " + token).replace(" ##", "")
        return token

    def get_input_embedding(self, token_id) -> Tensor:
        return self.get_input_embeddings([token_id])[0]

    def get_input_embeddings(self, encoding) -> Tensor:
        return self.get_batch_input_embeddings([encoding])[0]

    def get_batch_input_embeddings(self, encodings) -> Tensor:
        with torch.no_grad():
            return self.model.get_input_embeddings()(torch.tensor(encodings))

    def get_average_vocab_probs(self, clear_embeddings, target_index):
        vocab_probs = torch.zeros(self.get_vocabulary_size(), dtype=torch.float32)
        for i in range(self.iteration_count):
            masked_embeddings = self.mask_target(clear_embeddings, target_index, self.dropout_rate, i)
            masked_output = self.get_output_from_embeddings(masked_embeddings)
            vocab_probs += self.get_vocab_probs(masked_output, 0, target_index)
        return vocab_probs / self.iteration_count

    def get_vocab_probs(self, output, text_index, target_index) -> Tensor:
        return torch.softmax(output.logits[text_index][target_index], dim=0)

    def mask_target(self, embeddings, target_index, dropout_rate, iteration_index):
        embeddings_copy = embeddings.clone()
        self.apply_dropout(embeddings_copy[target_index], dropout_rate, iteration_index)
        return embeddings_copy

    def apply_dropout(self, embedding: Tensor, dropout_rate, iteration_index):
        embedding_length = embedding.shape[0]
        dropout_count = round(dropout_rate * embedding_length)
        generator = Generator(device=embedding.device).manual_seed(iteration_index) if self.deterministic else None
        dropout_indices = torch.randperm(embedding_length, generator=generator)[:dropout_count]
        embedding[dropout_indices] = self.mask_embedding[dropout_indices] if self.use_mask_token else 0.0

    def get_alternatives_token_similarities(self, original_output, alternatives_output, layer_count=4) -> Tensor:
        tokens_alternative_similarities = []
        token_count = original_output.hidden_states[0].shape[1]
        for token_index in range(token_count):
            tokens_alternative_similarities.append(self.get_alternative_similarities_for_token(
                original_output, alternatives_output, token_index, layer_count))
        return torch.stack(tokens_alternative_similarities).t()

    def get_alternative_similarities_for_token(self, original_output, alternatives_output, token_index, layer_count=4):
        original_representation = self.get_representations(original_output, token_index, layer_count)
        alternative_representations = self.get_representations(alternatives_output, token_index, layer_count)
        return self.cos_similarity(original_representation, alternative_representations)

    def get_representations(self, output, token_index, layer_count=4) -> Tensor:
        return torch.cat(tuple([output.hidden_states[i][:, token_index, :] for i in range(-layer_count, 0)]), dim=1)

    def get_normalized_probs(self, candidate_probs: Tensor, original_prediction_prob) -> Tensor:
        return torch.div(candidate_probs, (1.0 - original_prediction_prob))

    def find_alternative_encodings(self, encoding, target_index, token_ids) -> Tensor:
        alternative_encodings = []
        for token_id in token_ids:
            new_encoding = encoding.copy()
            new_encoding[target_index] = token_id
            alternative_encodings.append(torch.tensor(new_encoding, dtype=torch.long))
        return torch.stack(alternative_encodings)

    def get_average_attention_matrix(self, output) -> Tensor:
        return torch.stack(output.attentions).squeeze(1).mean(0).mean(0)
