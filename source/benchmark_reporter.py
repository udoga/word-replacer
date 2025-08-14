import pandas as pd
import csv
import re
from nltk import WordNetLemmatizer
from statistics import mean
from external.generalized_average_precision import GeneralizedAveragePrecision
from source.substitution_request import SubstitutionRequest

class BenchmarkReporter:
    def __init__(self, dataset_folder, report_path=None, provide_candidates=False, print_tables=False,
            print_progress=False):
        self.frame = None
        self.dataset_folder = dataset_folder
        self.report_file = open(report_path, 'w', encoding='utf-8') if report_path else None
        self.provide_candidates = provide_candidates
        self.print_tables = print_tables
        self.print_progress = print_progress
        self.lemmatizer = WordNetLemmatizer()
        self.non_ascii_regex = re.compile(r'[^\x00-\x7F]+')

    def get_frame(self) -> pd.DataFrame:
        return self.frame

    def load_dataset(self, file_prefix, shuffle=False, row_count=None):
        self.load_sentences(file_prefix + ".preprocessed")
        self.load_gold_map(file_prefix + ".gold")
        if shuffle: self.frame = self.frame.sample(frac=1)
        if row_count: self.frame = self.frame.head(row_count)

    def load_sentences(self, file_name):
        self.frame = pd.read_csv(str(self.dataset_folder / file_name), names=["target", "id", "position", "text"],
                                 sep="\t", header=None, encoding="iso-8859-1", engine="python", quoting=csv.QUOTE_NONE)
        self.frame = self.frame.set_index("id")
        self.frame["text"] = self.frame["text"].apply(lambda s: self.non_ascii_regex.sub('', s))
        self.frame[['target', 'tag']] = (self.frame['target'].str.split('.', n=1, expand=True))
        self.frame['gold_map'] = [{} for _ in range(len(self.frame))]

    def load_gold_map(self, file_name):
        with open(str(self.dataset_folder / file_name), 'r', encoding="iso-8859-1") as file:
            for line in file:
                left, right = [p.strip() for p in line.strip().split("::")]
                idx = int(left.split()[-1])
                gold_map = {" ".join(c.split()[:-1]): int(c.split()[-1]) for c in right.split(";") if c.strip()}
                gold_map = {k: v for k, v in gold_map.items() if ' ' not in k}
                self.frame.at[idx, "gold_map"] = gold_map
                self.frame.at[idx, "tie"] = len(self.get_top_golds(gold_map)) > 1
        self.frame = self.frame[self.frame['gold_map'].map(bool)]

    def load_predictions(self, substituter):
        self.frame["predictions"] = self.frame.apply(
            lambda r: self.get_predictions(r.name, substituter, r.text, r.target, r.position, r.tag), axis=1)
        print()

    def load_scores(self):
        for r in self.frame.itertuples():
            predictions = [self.lemmatizer.lemmatize(p, r.tag.split(".")[-1]) for p in r.predictions]
            best_prediction = predictions[0] if predictions else ""
            top_golds = self.get_top_golds(r.gold_map)
            self.frame.at[r.Index, "best_score"] = self.get_vote_weight(best_prediction, r.gold_map)
            self.frame.at[r.Index, "best_mode_score"] = int(best_prediction in top_golds)
            self.frame.at[r.Index, "oot_score"] = sum([self.get_vote_weight(p, r.gold_map) for p in predictions])
            self.frame.at[r.Index, "oot_mode_score"] = int(any(p in top_golds for p in predictions))
            self.frame.at[r.Index, "precision@1"] = self.get_precision(predictions, 1, r.gold_map)
            self.frame.at[r.Index, "precision@3"] = self.get_precision(predictions, 3, r.gold_map)
            self.frame.at[r.Index, "recall@10"] = mean([int(s in predictions) for s in r.gold_map])
            self.frame.at[r.Index, "gap_score"] = self.get_gap_score(predictions, r.gold_map)

    def get_average_scores(self):
        frame_predicted = self.frame[self.frame["predictions"].map(bool)]
        frame_non_tie = frame_predicted[frame_predicted["tie"] == False]
        return {
            "best_score": frame_predicted["best_score"].mean().item(),
            "best_mode_score": frame_non_tie["best_mode_score"].mean().item(),
            "oot_score": frame_predicted["oot_score"].mean().item(),
            "oot_mode_score": frame_non_tie["oot_mode_score"].mean().item(),
            "precision@1": frame_predicted["precision@1"].mean().item(),
            "precision@3": frame_predicted["precision@3"].mean().item(),
            "recall@10": frame_predicted["recall@10"].mean().item(),
            "gap_score": frame_predicted["gap_score"].mean().item(),
        }

    def print_report(self):
        print("\nFinal scores:", file=self.report_file)
        for label, score in self.get_average_scores().items():
            print(f"{label:>20}: {score:.6f}", file=self.report_file)
        print("\nBenchmark table:", file=self.report_file)
        print(self.get_frame().to_string(max_colwidth=120), file=self.report_file)
        print("Benchmark report: " + self.report_file.name)

    def get_predictions(self, idx, substituter, text, target, position, tag):
        if self.print_progress: print(f"\rLoading predictions: {idx}/{self.frame.iloc[-1].name} ", end='', flush=True)
        try:
            candidates = self.get_gold_candidates(target) if self.provide_candidates else None
            table = substituter.substitute(SubstitutionRequest(text, target, position, tag, candidates))
            if self.print_tables: print(f"Id={idx} Target={target} Position={position} Text={text}\n{table}\n")
            return list(table)[:10]
        except Exception as e:
            print(f"Skipping record {idx}: {e}", file=self.report_file)
        return []

    def get_vote_weight(self, prediction, gold_map):
        return gold_map.get(prediction, 0) / sum(gold_map.values())

    def get_top_golds(self, gold_map):
        return [s for s, count in gold_map.items() if count == max(gold_map.values())]

    def get_precision(self, predictions, count, gold_map):
        return sum([int(p in gold_map) for p in predictions[:count]]) / count

    def get_gap_score(self, predictions, gold_map):
        gold_pairs = [[k, v] for k, v in gold_map.items()]
        prediction_pairs = [[predictions[i], len(predictions)-i] for i in range(len(predictions))]
        return GeneralizedAveragePrecision.calc(gold_pairs, prediction_pairs)

    def get_gold_candidates(self, target_word):
        gold_maps = self.frame[self.frame["target"] == target_word]["gold_map"].to_list()
        gold_candidates = [candidate for gold_map in gold_maps for candidate in gold_map.keys()]
        return sorted(list(set(gold_candidates)))
