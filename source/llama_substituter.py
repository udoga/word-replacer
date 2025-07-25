import re
from llama_cpp import Llama
from source.substitution_request import SubstitutionRequest
from typing import List

class LlamaSubstituter:
    def __init__(self):
        self.llm=Llama.from_pretrained(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            seed=0,
            chat_format="llama-3",
            verbose=False)

    def substitute(self, r: SubstitutionRequest) -> list:
        target_word = r.text.split()[r.position]
        messages = [
            {"role": "system", "content": "List 10 common words that can replace the target word best." },
            {"role": "user", "content": f'Sentence: "{r.text}"\nTarget word: "{target_word}"\nPosition: {r.position}'}]
        output = self.llm.create_chat_completion(messages=messages, temperature=0.0)
        response = output["choices"][0]["message"]["content"]
        return self.extract_word_list(response)

    def extract_word_list(self, raw: str) -> List[str]:
        pattern = re.compile(r'^\s*\d+\s*[\.\)\-]\s*(.+)$')
        return list(dict.fromkeys([
            m.group(1).strip().lower()
            for line in raw.splitlines()
            if (m := pattern.match(line))
        ]))
