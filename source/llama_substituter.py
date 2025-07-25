from llama_cpp import Llama
from source.substitution_request import SubstitutionRequest

class LlamaSubstituter:
    def __init__(self):
        self.llm=Llama.from_pretrained(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=4096,
            n_gpu_layers=-1,
            seed=0,
            chat_format="llama-3",
            verbose=False)

    def substitute(self, r: SubstitutionRequest) -> list:
        words = r.text.split()
        target_word = words[r.position]
        messages = [
            {"role": "system", "content": "Return 10 top single-word substitutes in JSON {\"candidates\": []}"},
            {"role": "user", "content": f'Sentence: "{r.text}"\nTarget word: "{target_word}"\nPosition: {r.position}'}
        ]
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"})
        return out["choices"][0]["message"]["content"]
