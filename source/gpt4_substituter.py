import json
from openai import OpenAI
from candidate_excluder import CandidateExcluder
from substitution_request import SubstitutionRequest
from substitution_table import SubstitutionTable

class Gpt4Substituter:
    def __init__(self, model="gpt-4o", temperature=0.7):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.excluder = CandidateExcluder()

    def substitute(self, r: SubstitutionRequest):
        words = r.text.split()
        target_word = words[r.position]
        assert self.excluder.has_same_root(r.target, target_word), f"Target {r.target} != {target_word} at {r.position}"
        candidates = self.rank_candidates(r, target_word) if r.candidates else self.propose_candidates(r, target_word)
        ranks = list(range(len(candidates), 0, -1))
        return SubstitutionTable({"candidate": candidates, "gpt4_rank": ranks})

    def propose_candidates(self, r: SubstitutionRequest, target_word):
        messages = [
            {"role": "system", "content": "Return 10 top single-word substitutes in JSON {\"candidates\": []}"},
            {"role": "user", "content": f'Sentence: "{r.text}"\nTarget word: "{target_word}"\nPosition: {r.position}'}]
        return self.get_candidates(messages)

    def rank_candidates(self, r: SubstitutionRequest, target_word):
        messages = [
            {"role": "system", "content": "Sort candidates by substitution quality. Return JSON {\"candidates\": []}"},
            {"role": "user", "content": f'Sentence: "{r.text}"\nTarget word: "{target_word}"\nPosition: {r.position}'
                                        f'\nCandidates: {r.candidates}'}]
        return self.get_candidates(messages)

    def get_candidates(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=messages)
        return json.loads(response.choices[0].message.content)["candidates"]
