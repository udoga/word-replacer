from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM
from source.dropout_substituter import DropoutSubstituter
from source.benchmark_reporter import BenchmarkReporter

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', add_prefix_space=True)
model = BertForMaskedLM.from_pretrained('bert-large-uncased', output_hidden_states=True, output_attentions=True, attn_implementation="eager")
substituter = DropoutSubstituter(tokenizer, model, dropout_rate=0.3, candidate_count=50, alpha=0.01, iteration_count=5, deterministic=True)

def run_substituter():
    text = "The wine he sent to me as my birthday gift is too strong to drink."
    table = substituter.substitute(text, target="strong", position=12)
    print(table)

def run_benchmark():
    reporter = BenchmarkReporter(Path(__file__).resolve().parents[0] / "dataset")
    reporter.load_dataset("lst_trial", 20)
    reporter.load_predictions(substituter)
    reporter.load_scores()
    print(reporter.get_frame().to_string(max_colwidth=30))
    print("\nFinal scores:", reporter.get_average_scores())

run_substituter()
