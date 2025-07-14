import torch
from pandas import DataFrame
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GptSubstituter:
    def __init__(self, model_name):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def substitute(self, text, target, position):
        words = text.split()
        text_before = " ".join(words[:position])
        prompt = text + " " + text_before
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction_logits = outputs.logits[0, -1, :]
        prediction_probs = torch.softmax(prediction_logits, dim=0)
        top_probs, top_ids = torch.topk(prediction_probs, k=50)
        candidates = [self.tokenizer.decode([token_id]).strip() for token_id in top_ids]
        return DataFrame({'candidate': candidates, 'probability': top_probs.cpu()})
