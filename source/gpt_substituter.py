import torch
from pandas import DataFrame
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from source.candidate_excluder import CandidateExcluder
from source.substitution_table import SubstitutionTable

class GptSubstituter:
    def __init__(self, model_name):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).eval()
        self.excluder = CandidateExcluder()

    def substitute(self, text, target, position) -> SubstitutionTable:
        words = text.split()
        text_before = " ".join(words[:position])
        prompt = text + " I repeat. " + text_before
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction_logits = outputs.logits[0, -1, :]
        prediction_probs = torch.softmax(prediction_logits, dim=0)
        top_probs, top_ids = torch.topk(prediction_probs, k=50)
        candidate_tokens = [self.tokenizer.decode([token_id]) for token_id in top_ids]
        frame = DataFrame()
        frame['candidate'] = [t.strip() for t in candidate_tokens]
        frame['is_included'] = self.excluder.are_candidates_included(candidate_tokens, target)
        frame['probability'] = top_probs.cpu()
        return SubstitutionTable.from_frame(frame.set_index('candidate'))
