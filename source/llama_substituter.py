import re
import pandas as pd
import llama_cpp
from typing import List
from source.candidate_excluder import CandidateExcluder
from source.substitution_request import SubstitutionRequest
from source.substitution_table import SubstitutionTable
from importlib.util import find_spec
from pathlib import Path

class LlamaSubstituter:
    def __init__(self, proposer=None, target_similarity_enabled=False, sentence_similarity_enabled=False):
        self.proposer = proposer
        self.target_similarity_enabled = target_similarity_enabled
        self.sentence_similarity_enabled = sentence_similarity_enabled
        self.excluder = CandidateExcluder()
        self.lib = llama_cpp.load_shared_library("llama", Path(find_spec("llama_cpp").origin).parent / "lib")
        self.model = llama_cpp.Llama.from_pretrained(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            seed=0,
            chat_format="llama-3",
            verbose=False,
            embedding=True)
        print("GPU offload support:", bool(self.lib.llama_supports_gpu_offload()))

    def substitute(self, r: SubstitutionRequest) -> SubstitutionTable:
        target_word = r.text.split()[r.position]
        candidates = [target_word] + self.get_candidates(r, target_word)
        is_included = [False] + [True] * (len(candidates)-1)
        candidate_sentences = self.get_candidate_sentences(r.text, r.position, candidates)
        target_similarities = self.get_target_similarities(r.text, r.position, candidate_sentences)
        sentence_similarities = self.get_sentence_similarities(r.text, r.position, candidate_sentences)
        return self.get_substitution_table(candidates, is_included, target_similarities, sentence_similarities)

    def get_candidates(self, r: SubstitutionRequest, target_word) -> List[str]:
        if r.candidates: return r.candidates
        if self.proposer: return list(self.proposer.substitute(r))
        return self.propose_candidates(r, target_word)

    def propose_candidates(self, r: SubstitutionRequest, target_word) -> List[str]:
        messages = [
            {"role": "system", "content": "List 10 common words that can replace the target word best."},
            {"role": "user", "content": f'Sentence: "{r.text}"\nTarget word: "{target_word}"\nPosition: {r.position}'}]
        self.set_llama_embeddings(False)
        output = self.model.create_chat_completion(messages=messages, temperature=0.0)
        response = output["choices"][0]["message"]["content"]
        word_list = self.extract_word_list(response)
        return list(dict.fromkeys([word.split()[-1] for word in word_list]))

    def get_substitution_table(self, candidates, is_included, target_similarities, sentence_similarities):
        return SubstitutionTable.from_frame(pd.DataFrame({
                "candidate": candidates,
                "is_included": is_included,
                "target_similarity": target_similarities,
                "sentence_similarity": sentence_similarities
        }).sort_values(by=["sentence_similarity"], ascending=False))

    def extract_word_list(self, response: str) -> List[str]:
        pattern = re.compile(r'^\s*\d+\s*[\.\)\-]\s*(.+)$')
        return [
            m.group(1).strip().lower()
            for line in response.splitlines()
            if (m := pattern.match(line))]

    def get_candidate_sentences(self, text, position, candidates):
        candidate_sentences = []
        for candidate in candidates:
            words = text.split()
            words[position] = candidate
            candidate_sentences.append(" ".join(words))
        return candidate_sentences

    def get_target_similarities(self, text, position, candidate_sentences):
        if not self.target_similarity_enabled: return [0.0] * len(candidate_sentences)
        original_target_embedding = self.get_target_embedding(text, position)
        candidate_target_embeddings = [self.get_target_embedding(s, position) for s in candidate_sentences]
        similarities = [self.get_similarity(original_target_embedding, e) for e in candidate_target_embeddings]
        return self.normalise(similarities)

    def get_similarity(self, a: List[float], b: List[float]) -> float:
        assert len(a) == len(b), f"Different embedding lengths: original={len(a)} candidate={len(b)}"
        return sum(x * y for x, y in zip(a, b))

    def normalise(self, scores):
        return [s / scores[0] for s in scores]

    def get_first_token_index(self, text: str, position: int) -> int:
        start = list(re.finditer(r"\S+", text))[position].start()
        encoding = self.model.tokenize(text[:start].encode("utf-8"), add_bos=True)
        return len(encoding) - 1

    def get_last_token_index(self, text: str, position: int) -> int:
        end = list(re.finditer(r"\S+", text))[position].end()
        encoding = self.model.tokenize(text[:end].encode("utf-8"), add_bos=True)
        return len(encoding) - 1

    def get_token_embeddings(self, text):
        self.set_llama_embeddings(True)
        return self.model.create_embedding(input=[text])["data"][0]["embedding"]

    def get_target_embedding(self, text: str, position: int):
        return self.get_token_embeddings(text)[self.get_last_token_index(text, position)]

    def get_sentence_similarities(self, text, position, candidate_sentences):
        if not self.sentence_similarity_enabled: return [0.0] * len(candidate_sentences)
        original_embedding = self.get_sentence_embedding(text, position)
        similarities = [self.get_sentence_similarity(s, position, original_embedding) for s in candidate_sentences]
        return self.normalise(similarities)

    def get_sentence_embedding(self, text, position):
        token_embeddings = self.get_embeddings_with_one_from_position(text, position)
        return self.concatenate_embeddings(token_embeddings)

    def get_sentence_similarity(self, candidate_text, position, original_embedding):
        candidate_embedding = self.get_sentence_embedding(candidate_text, position)
        return self.get_similarity(original_embedding, candidate_embedding)

    def get_embeddings_with_one_from_position(self, text, position):
        embeddings = self.get_token_embeddings(text)
        first_token_index = self.get_first_token_index(text, position)
        last_token_index = self.get_last_token_index(text, position)
        return embeddings[0:first_token_index] + embeddings[last_token_index:]

    def concatenate_embeddings(self, embeddings):
        return [x for embedding in embeddings for x in embedding]

    def set_llama_embeddings(self, flag: bool):
        llama_cpp.llama_cpp.llama_set_embeddings(self.model._ctx.ctx, flag)
