import pandas as pd
import csv

class BenchmarkReporter:
    def __init__(self, dataset_folder):
        self.frame = None
        self.dataset_folder = dataset_folder

    def get_frame(self) -> pd.DataFrame:
        return self.frame

    def load_dataset(self, file_prefix, row_count = None):
        self.load_sentences(file_prefix + ".preprocessed")
        self.load_substitutes(file_prefix + ".gold")
        if row_count: self.frame = self.frame.head(row_count)

    def load_sentences(self, file_name):
        self.frame = pd.read_csv(str(self.dataset_folder / file_name), names=["target", "id", "position", "text"],
                                 sep="\t", header=None, encoding="iso-8859-1", engine="python", quoting=csv.QUOTE_NONE)
        self.frame = self.frame.set_index("id")
        self.frame[['target', 'type']] = (self.frame['target'].str.split('.', n=1, expand=True))
        self.frame['substitutes'] = [{} for _ in range(len(self.frame))]

    def load_substitutes(self, file_name):
        with open(str(self.dataset_folder / file_name), 'r', encoding="iso-8859-1") as file:
            for line in file:
                left, right = [p.strip() for p in line.strip().split("::")]
                idx = int(left.split()[-1])
                substitutes = {" ".join(c.split()[:-1]): int(c.split()[-1]) for c in right.split(";") if c.strip()}
                self.frame.at[idx, "substitutes"] = substitutes

    def load_predictions(self, substituter):
        self.frame["predictions"] = self.frame.apply(
            lambda r: self.get_ten_predictions(substituter, r["text"], r["target"],  r["position"]), axis=1)

    def load_scores(self):
        self.frame[["best_score", "best_mode_score", "oot_score", "oot_mode_score"]] = 0.0
        for r in self.frame.itertuples():
            best_prediction = r.predictions[0]
            modes = [sub for sub, count in r.substitutes.items() if count == max(r.substitutes.values())]
            tie = len(modes) != 1
            self.frame.at[r.Index, "best_score"] = self.score_prediction(best_prediction, r.substitutes)
            self.frame.at[r.Index, "best_mode_score"] = None if tie else int(best_prediction in modes)
            self.frame.at[r.Index, "oot_score"] = sum([self.score_prediction(p, r.substitutes) for p in r.predictions])
            self.frame.at[r.Index, "oot_mode_score"] = None if tie else int(any(p in modes for p in r.predictions))

    def get_ten_predictions(self, substituter, text, target, position):
        predictions = []
        try:
            predictions = substituter.get_predictions(text, target, position)
        except Exception as e:
            print(f"Could not get predictions: text='{text}' target='{target}' position='{position}'' error={e}")
        return (predictions + ["" for _ in range(max(0, 10 - len(predictions)))])[:10]

    def score_prediction(self, prediction, substitute_map):
        return substitute_map.get(prediction, 0) / sum(substitute_map.values())
