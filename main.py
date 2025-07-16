import torch
from pathlib import Path
from source.gpt_substituter import GptSubstituter
from source.pattern_substituter import PatternSubstituter
from source.bert_substituter import BertSubstituter
from source.benchmark_reporter import BenchmarkReporter

def run_substituter(substituter):
    text = "The wine he sent to me as my birthday gift is too strong to drink."
    table = substituter.substitute(text, target="strong", position=12)
    print(table)

def run_benchmark(substituter):
    reporter = BenchmarkReporter(Path(__file__).resolve().parents[0] / "dataset", print_progress=True)
    reporter.load_dataset("lst_trial")
    reporter.load_predictions(substituter)
    reporter.load_scores()
    reporter.print_report()

def get_substituter(name):
    if name == "dropout": return BertSubstituter("roberta-base", dropout_rate=0.3, iteration_count=5, alpha=0.01)
    if name == "concat": return BertSubstituter("roberta-base", concatenate=True, dropout_rate=1, use_mask_token=True)
    if name == "pattern": return PatternSubstituter(get_substituter("dropout"), ["%", "or", "%"], position_change=2)
    if name == "gpt": return GptSubstituter("gpt2-large", pll_enabled=False)
    return None

torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
run_substituter(get_substituter("dropout"))
