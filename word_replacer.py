#!/usr/bin/env python3

import torch
import sys
from pathlib import Path
from source.gpt2_substituter import Gpt2Substituter
from source.gpt4_substituter import Gpt4Substituter
from source.pattern_substituter import PatternSubstituter
from source.bart_substituter import BartSubstituter
from source.bert_substituter import BertSubstituter
from source.benchmark_reporter import BenchmarkReporter
from source.substitution_request import SubstitutionRequest
from source.llama_substituter import LlamaSubstituter

def run_demo(substituter):
    run_substituter(substituter, "The wine he sent to me as my birthday gift is too strong to drink.", "strong", 12)

def run_substituter(substituter, text, target, position):
    print("Text:", text, "\nTarget:", target, "\nPosition:", position)
    print("Substitution Table:\n", substituter.substitute(SubstitutionRequest(text, target, position)))

def run_benchmark(substituter, dataset_name, report_path):
    reporter = BenchmarkReporter(Path(__file__).resolve().parents[0] / "dataset", report_path, print_progress=True)
    reporter.load_dataset(dataset_name)
    reporter.load_predictions(substituter)
    reporter.load_scores()
    reporter.print_report()

def get_substituter(name):
    torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if name == "zhou": return BertSubstituter("bert-large-uncased", iteration_count=5, score_basis="validation_score")
    if name == "dropout": return BertSubstituter("roberta-base", dropout_rate=0.3, iteration_count=5, alpha=0.01)
    if name == "concat": return BertSubstituter("roberta-base", concatenate=True, dropout_rate=1, use_mask_token=True)
    if name == "pattern": return PatternSubstituter(get_substituter("dropout"), ["%", "or", "%"], position_change=2)
    if name == "gpt2": return Gpt2Substituter("gpt2-large", pll_enabled=False)
    if name == "gpt4": return Gpt4Substituter("gpt-4o", temperature=0.7)
    if name == "llama": return LlamaSubstituter(target_similarity_enabled=False, sentence_similarity_enabled=False)
    if name == "bart": return BartSubstituter(proposer=get_substituter("llama"))
    raise Exception("Unknown method:", name)

def print_usage():
    print("Usage: word_replacer.py <Function> [Arguments]")
    print("Functions:")
    print("- demo <method>")
    print("- substitute <method> <text> <target> <position>")
    print("- benchmark <method> <dataset> [report_path]")
    print("Methods: zhou, dropout, concat, pattern, gpt2, gpt4, llama, bart")
    print("Datasets: lst_trial, lst_test, lst_all, coinco_trial, coinco_test, coinco_all")

def run(s):
    if len(s) == 3 and s[1] == "demo": run_demo(get_substituter(s[2]))
    elif len(s) == 6 and s[1] == "substitute": run_substituter(get_substituter(s[2]), s[3], s[4], int(s[5]))
    elif len(s) >= 4 and s[1] == "benchmark": run_benchmark(get_substituter(s[2]), s[3], None if len(s) <= 4 else s[4])
    else: print_usage()

run(sys.argv)
