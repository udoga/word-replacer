import math
import re
import pandas as pd
from llama_cpp import Llama
from llama_cpp import llama_cpp as low
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
        candidates = r.candidates if r.candidates else self.get_candidates(r)
        candidate_sentences = self.get_candidate_sentences(r.text, r.position, candidates)
        similarities = self.get_sentence_similarities(r.text, r.position, candidate_sentences)
        df = pd.DataFrame({"candidate": candidates, "similarity": similarities})
        return SubstitutionTable.from_frame(df.sort_values(by=["similarity"], ascending=False))

    def get_candidates(self, r: SubstitutionRequest) -> List[str]:
        target_word = r.text.split()[r.position]
        messages = [
            {"role": "system", "content": "List 10 common words that can replace the target word best." },
            {"role": "user", "content": f'Sentence: "{r.text}"\nTarget word: "{target_word}"\nPosition: {r.position}'}]
        self.set_llama_embeddings(False)
        output = self.llm.create_chat_completion(messages=messages, temperature=0.0)
        response = output["choices"][0]["message"]["content"]
        return self.extract_word_list(response)

    def extract_word_list(self, response: str) -> List[str]:
        pattern = re.compile(r'^\s*\d+\s*[\.\)\-]\s*(.+)$')
        return list(dict.fromkeys([
            m.group(1).strip().lower()
            for line in response.splitlines()
            if (m := pattern.match(line))
        ]))

    def get_candidate_sentences(self, text, position, candidates):
        candidate_sentences = []
        for candidate in candidates:
            words = text.split()
            words[position] = candidate
            candidate_sentences.append(" ".join(words))
        return candidate_sentences

    def get_target_similarities(self, text, position, sentences):
        original_embedding = self.get_word_embedding(text, position)
        return [self.get_similarity(original_embedding, self.get_word_embedding(s, position)) for s in sentences]

    def get_similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def get_word_embedding(self, text: str, position: int):
        return self.get_normalised_token_embedding(text, self.get_last_token_index(text, position))

    def get_first_token_index(self, text: str, position: int) -> int:
        start = list(re.finditer(r"\S+", text))[position].start()
        encoding = self.llm.tokenize(text[:start].encode("utf-8"), add_bos=True)
        return len(encoding) - 1

    def get_last_token_index(self, text: str, position: int) -> int:
        end = list(re.finditer(r"\S+", text))[position].end()
        encoding = self.llm.tokenize(text[:end].encode("utf-8"), add_bos=True)
        return len(encoding) - 1

    def get_token_embeddings(self, text):
        self.set_llama_embeddings(True)
        return self.llm.create_embedding(input=[text])["data"][0]["embedding"]

    def get_normalised_token_embedding(self, text: str, token_index: int):
        return self.normalise_embedding(self.get_token_embeddings(text)[token_index])

    def normalise_embedding(self, embedding):
        normalizer = math.sqrt(sum(e * e for e in embedding)) or 1.0
        return [e / normalizer for e in embedding]

    def get_sentence_similarities(self, text, position, candidate_sentences):
        original_embeddings = self.get_embeddings_with_one_from_position(text, position)
        original_concatenated_embedding = self.concatenate_embeddings(original_embeddings)
        return [self.get_sentence_similarity(s, position, original_concatenated_embedding) for s in candidate_sentences]

    def get_sentence_similarity(self, candidate_text, position, original_concatenated_embedding):
        candidate_embeddings = self.get_embeddings_with_one_from_position(candidate_text, position)
        candidate_concatenated_embedding = self.concatenate_embeddings(candidate_embeddings)
        return self.get_similarity(original_concatenated_embedding, candidate_concatenated_embedding)

    def get_embeddings_with_one_from_position(self, text, position):
        embeddings = self.get_token_embeddings(text)
        first_token_index = self.get_first_token_index(text, position)
        last_token_index = self.get_last_token_index(text, position)
        return embeddings[0:first_token_index] + embeddings[last_token_index:]

    def concatenate_embeddings(self, embeddings):
        return [x for embedding in embeddings for x in embedding]

    def set_llama_embeddings(self, flag: bool):
        low.llama_set_embeddings(self.llm._ctx.ctx, flag)
