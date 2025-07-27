import math
import re
from llama_cpp import Llama
from source.substitution_request import SubstitutionRequest
from typing import List
from substitution_table import SubstitutionTable

class LlamaSubstituter:
    def __init__(self):
        self.llm=Llama.from_pretrained(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            seed=0,
            chat_format="llama-3",
            verbose=False,
            embedding=True)

    def substitute(self, r: SubstitutionRequest) -> SubstitutionTable:
        words = r.text.split()
        target_word = words[r.position]
        messages = [
            {"role": "system", "content": "List 10 common words that can replace the target word best." },
            {"role": "user", "content": f'Sentence: "{r.text}"\nTarget word: "{target_word}"\nPosition: {r.position}'}]
        output = self.llm.create_chat_completion(messages=messages, temperature=0.0)
        response = output["choices"][0]["message"]["content"]
        candidates = self.extract_word_list(response)
        candidate_sentences = self.get_candidate_sentences(words, r.position, candidates)
        target_similarities = self.get_target_similarities(r.text, r.position, candidate_sentences)
        return SubstitutionTable({"candidate": candidates, "similarity": target_similarities})

    def extract_word_list(self, response: str) -> List[str]:
        pattern = re.compile(r'^\s*\d+\s*[\.\)\-]\s*(.+)$')
        return list(dict.fromkeys([
            m.group(1).strip().lower()
            for line in response.splitlines()
            if (m := pattern.match(line))
        ]))

    def get_candidate_sentences(self, words, position, candidates):
        candidate_sentences = []
        for candidate in candidates:
            tokens = words.copy()
            tokens[position] = candidate
            candidate_sentences.append(" ".join(tokens))
        return candidate_sentences

    def get_target_similarities(self, text, position, sentences):
        original_embedding = self.get_word_embedding(text, position)
        return [self.get_cos_similarity(original_embedding, self.get_word_embedding(s, position)) for s in sentences]

    def get_cos_similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def get_word_embedding(self, text: str, position: int):
        token_index = self.get_token_index_from_position(text, position)
        return self.get_token_embedding(text, token_index)

    def get_token_index_from_position(self, text: str, position: int) -> int:
        end_index = list(re.finditer(r"\S+", text))[position].end()
        encoding = self.llm.tokenize(text[:end_index].encode("utf-8"), add_bos=True)
        return len(encoding) - 1

    def get_token_embedding(self, text: str, token_index: int):
        embedding = self.llm.create_embedding(input=[text])["data"][0]["embedding"][token_index]
        normalizer = math.sqrt(sum(e * e for e in embedding)) or 1.0
        return [e / normalizer for e in embedding]
