from transformers import RobertaTokenizer, RobertaForMaskedLM
from source.lemmatizer import Lemmatizer
from source.dropout_substituter import DropoutSubstituter
from source.benchmark_reporter import BenchmarkReporter

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True, output_attentions=True, attn_implementation="eager")
lemmatizer = Lemmatizer()
substituter = DropoutSubstituter(tokenizer, model, lemmatizer, dropout_rate=0.3, candidate_count=50, alpha=0.01, iteration_count=1, deterministic=True)

def run_substituter():
    print("Running substituter...")
    text = "The wine he sent to me as my birthday gift is too powerful to drink"
    target = "powerful"
    table = substituter.substitute(text, target, 12)
    print(table)

def run_benchmark():
    print("Running benchmark...")
    benchmark_reporter = BenchmarkReporter("dataset/")
    benchmark_reporter.load_dataset("lst_test", 20)
    benchmark_reporter.load_predictions(substituter)
    benchmark_reporter.load_scores()
    print(benchmark_reporter.get_frame().to_string())

run_benchmark()
