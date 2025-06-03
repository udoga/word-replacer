import numpy as np
import torch

from torch import Tensor
from transformers import RobertaTokenizer, RobertaForMaskedLM

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
        candidate_ids = torch.topk(prediction_probs, k=self.candidate_count, dim=0).indices
        candidate_probs = prediction_probs[candidate_ids].tolist()
        candidates = [c.strip() for c in self.tokenizer.batch_decode(candidate_ids)]
        return dict(zip(candidates, candidate_probs))

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
        dropout_indices = np.random.choice(embedding_length, dropout_count, replace=False)
        embedding[dropout_indices] = 0

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True)
substituter = DropoutSubstituter(model, tokenizer, 0.5, 10)

text = "The wine he sent to me as my birthday gift is too powerful to drink."
target = "powerful"
results = substituter.substitute(text, target)
print(results)
