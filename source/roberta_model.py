import torch
from torch import Tensor
from transformers import RobertaTokenizer, RobertaForMaskedLM

class RobertaModel:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True)
        self.output = None

    def get_vocabulary_size(self):
        return len(self.tokenizer)

    def get_token_ids_from_text(self, text):
        return self.tokenizer.encode(" " + text)

    def get_tokens_from_ids(self, token_ids):
        return [c.strip() for c in self.tokenizer.batch_decode(token_ids)]

    def get_input_embeddings(self, encoding) -> Tensor:
        return self.get_batch_input_embeddings([encoding])[0]

    def get_batch_input_embeddings(self, encodings) -> Tensor:
        with torch.no_grad():
            return self.model.get_input_embeddings()(torch.tensor(encodings))

    def get_prediction_logits(self, input_embeddings, target_position) -> Tensor:
        with torch.no_grad():
            return self.model(inputs_embeds=input_embeddings.unsqueeze(0)).logits[0][target_position]

    def get_hidden_states(self) -> tuple:
        return self.output.hidden_states if self.output else None

    def load_encodings(self, encodings):
        with torch.no_grad():
            self.output = self.model(torch.tensor(encodings))

    def get_contextualized_representations(self, token_index, layer_count) -> Tensor:
        layer_indices = [-i for i in range(1, layer_count + 1)]
        return torch.cat(tuple([self.output.hidden_states[i][:, token_index, :] for i in layer_indices]), dim=1)
