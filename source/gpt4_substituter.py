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
        candidates = self.get_candidates(r.text, target_word, r.position)
        ranks = list(range(len(candidates), 0, -1))
        return SubstitutionTable({"candidate": candidates, "gpt4_rank": ranks})

    def get_candidates(self, text, target_word, position):
        messages = [
            {"role": "system", "content": "Return 10 top single-word substitutes in JSON {\"candidates\": []}"},
            {"role": "user", "content": f'Sentence: "{text}"\nTarget word: "{target_word}"\nPosition: {position}'}]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=messages)
        return json.loads(response.choices[0].message.content)["candidates"]
