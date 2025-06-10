import torch
from torch import Tensor
from transformers import RobertaTokenizer, RobertaForMaskedLM

class RobertaModel:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForMaskedLM.from_pretrained('roberta-base',
                                                        output_hidden_states=True,
                                                        output_attentions=True,
                                                        attn_implementation="eager")

    def get_output_from_encodings(self, encodings):
        with torch.no_grad():
            return self.model(encodings)

    def get_output_from_embeddings(self, embeddings):
        with torch.no_grad():
            return self.model(inputs_embeds=embeddings.unsqueeze(0))

    def get_vocabulary_size(self):
        return len(self.tokenizer)

    def get_encoding_from_text(self, text):
        return self.tokenizer.encode(" " + text)

    def get_tokens_from_ids(self, token_ids):
        return [c.strip() for c in self.tokenizer.batch_decode(token_ids)]

    def get_input_embeddings(self, encoding) -> Tensor:
        return self.get_batch_input_embeddings([encoding])[0]

    def get_batch_input_embeddings(self, encodings) -> Tensor:
        with torch.no_grad():
            return self.model.get_input_embeddings()(torch.tensor(encodings))
