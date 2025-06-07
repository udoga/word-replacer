import torch
from torch import Tensor

class OutputAnalyzer:
    def __init__(self, output):
        self.output = output

    def get_prediction_logits(self, text_index, target_index) -> Tensor:
        return self.output.logits[text_index][target_index]

    def get_contextualized_representations(self, token_index, layer_count) -> Tensor:
        layer_indices = [-i for i in range(1, layer_count + 1)]
        return torch.cat(tuple([self.output.hidden_states[i][:, token_index, :] for i in layer_indices]), dim=1)
