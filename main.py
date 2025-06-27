import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from source.bert_substituter import BertSubstituter
from source.benchmark_reporter import BenchmarkReporter

def run_substituter(substituter):
    text = "The wine he sent to me as my birthday gift is too powerful to drink."
    table = substituter.substitute(text, target="powerful", position=12)
    print(table)

def run_benchmark(substituter):
    reporter = BenchmarkReporter(Path(__file__).resolve().parents[0] / "dataset")
    reporter.load_dataset("lst_trial", 100)
    reporter.load_predictions(substituter)
    reporter.load_scores()
    print(reporter.get_frame().to_string(max_colwidth=100))
    print("\nFinal scores:", reporter.get_average_scores())

model_name = "roberta-base"
torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, do_lower_case=True, add_prefix_space=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True,
                                             attn_implementation="eager").to(torch.get_default_device())

dropout_substituter = BertSubstituter(tokenizer, model, dropout_rate=0.3, iteration_count=5, deterministic=True)
concat_substituter = BertSubstituter(tokenizer, model, concatenate=True, dropout_rate=1, alpha=0.01, candidate_count=50)

run_substituter(concat_substituter)
