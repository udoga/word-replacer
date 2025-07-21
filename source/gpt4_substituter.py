import json
from openai import OpenAI
from candidate_excluder import CandidateExcluder

class Gpt4Substituter:
    def __init__(self, model="gpt-4o", temperature=0.7):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.excluder = CandidateExcluder()

    def substitute(self, text, target, position, tag):
        words = text.split()
        target_word = words[position]
        assert self.excluder.has_same_root(target, target_word), f"Target {target} != {target_word} at {position}"
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return 10 top single-word substitutes in JSON {\"candidates\": []}"},
                {"role": "user", "content": f'Sentence: "{text}"\nTarget word: "{target_word}"\nPosition: {position}' }
            ]
        )
        return json.loads(response.choices[0].message.content)["candidates"]
