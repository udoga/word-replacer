import numpy as np
import torch

from transformers import RobertaTokenizer, RobertaForMaskedLM

class DropoutSubstituter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def substitute(self, text, target, dropout_rate):
        token_ids = self.tokenizer.encode(text)
        target_token_id = self.tokenizer.encode(" " + target)[1]
        target_position = token_ids.index(target_token_id)
        token_embeddings = self.get_token_embeddings(token_ids)
        self.apply_dropout(token_embeddings[target_position], dropout_rate)
        all_token_logits = self.get_logits_for_target(token_embeddings, target_position)
        all_token_probs = torch.softmax(all_token_logits, dim=0)
        top_token_ids = torch.topk(all_token_logits, k=10, dim=0).indices
        top_tokens = [tokenizer.decode(i.item()).strip() for i in top_token_ids]
        top_token_probs = all_token_probs[top_token_ids].tolist()
        return dict(zip(top_tokens, top_token_probs))

    def get_token_embeddings(self, token_ids):
        with torch.no_grad():
            return self.model(torch.tensor([token_ids])).hidden_states[0][0]

    def get_logits_for_target(self, input_embeddings, target_index):
        with torch.no_grad():
            model_output = model(inputs_embeds=input_embeddings.unsqueeze(0))
        logits_per_token = model_output.logits[0]
        return logits_per_token[target_index]

    def apply_dropout(self, embedding, dropout_rate):
        embedding_length = embedding.shape[0]
        dropout_count = round(dropout_rate * embedding_length)
        dropout_indices = np.random.choice(embedding_length, dropout_count, replace=False)
        embedding[dropout_indices] = 0

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True)
substituter = DropoutSubstituter(model, tokenizer)

text = "The wine he sent to me as my birthday gift is too powerful to drink."
target = "powerful"
results = substituter.substitute(text, target, 0.4)
print(results)
