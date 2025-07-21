import torch
from pandas import DataFrame
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from source.candidate_excluder import CandidateExcluder
from source.substitution_table import SubstitutionTable

class Gpt2Substituter:
    def __init__(self, model_name, pll_enabled=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(torch.get_default_device()).eval()
        self.excluder = CandidateExcluder()
        self.pll_enabled = pll_enabled

    def substitute(self, text, target, position, tag="") -> SubstitutionTable:
        words = text.split()
        text_before = " ".join(words[:position])
        target_index = self.find_token_index(text, position)
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]
        target_token = tokens[target_index].strip()
        assert self.excluder.has_same_root(target, target_token), f"Target {target} != {target_token} at {target_index}"
        prediction_probs = self.get_next_token_probabilities(text + " I repeat. " + text_before)
        candidate_probs, candidate_ids = torch.topk(prediction_probs, k=50)
        candidate_tokens = [self.tokenizer.decode([token_id]) for token_id in candidate_ids]
        candidates_included = self.excluder.are_candidates_included(candidate_tokens, target, tag)
        candidate_encodings = [self.get_candidate_encoding(text, position, c.strip()) for c in candidate_tokens]
        pll_scores = [self.get_pll_score(e) for e in candidate_encodings]
        frame = self.create_frame(candidate_tokens, candidates_included, candidate_probs, pll_scores)
        return SubstitutionTable.from_frame(frame)

    def get_next_token_probabilities(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction_logits = outputs.logits[0, -1, :]
        return torch.softmax(prediction_logits, dim=0)

    def create_frame(self, candidate_tokens, is_included, probabilities, pll_scores):
        frame = DataFrame()
        frame['candidate'] = [t.strip() for t in candidate_tokens]
        frame['is_included'] = is_included
        frame['probability'] = probabilities.cpu()
        frame['pll_score'] = pll_scores
        frame['probability_rank'] = frame['probability'].rank(ascending=True, method='dense').astype(int)
        frame['pll_rank'] = frame['pll_score'].rank(ascending=True, method='dense').astype(int)
        frame['rank'] = frame['probability_rank'] + frame['pll_rank']
        frame = frame.sort_values(by=["rank"], ascending=False)
        return frame.set_index('candidate')

    def get_candidate_encoding(self, text, position, candidate_token):
        words = text.split()
        words[position] = candidate_token
        words = words + text.split()
        candidate_text = " ".join(words)
        return self.tokenizer.encode(candidate_text, return_tensors='pt')

    def get_pll_score(self, encoding):
        if not self.pll_enabled: return 0.0
        with torch.no_grad():
            logits = self.model(encoding).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        text_log_prob = log_probs[0, :-1, :].gather(1, encoding[:, 1:].T).sum()
        return text_log_prob.item() / (encoding.size(1) - 1)

    def find_token_index(self, text, position):
        words = text.split()
        text_before = " ".join(words[:position+1])
        encoding = self.tokenizer.encode(text_before)
        return len(encoding) - 1
