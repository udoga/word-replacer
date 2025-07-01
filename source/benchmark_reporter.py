import pandas as pd
import csv
from nltk import WordNetLemmatizer

class BenchmarkReporter:
    def __init__(self, dataset_folder):
        self.frame = None
        self.dataset_folder = dataset_folder
        self.lemmatizer = WordNetLemmatizer()

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
                self.frame.at[idx, "tie"] = len(self.get_top_substitutes(substitutes)) > 1
        self.frame = self.frame[self.frame['substitutes'].map(bool)]

    def load_predictions(self, substituter):
        self.frame["predictions"] = self.frame.apply(
            lambda r: self.get_predictions(r.name, substituter, r["text"], r["target"], r["position"]), axis=1)
        print()

    def load_scores(self):
        self.frame[["best_score", "best_mode_score", "oot_score", "oot_mode_score"]] = 0.0
        for r in self.frame.itertuples():
            predictions = [self.lemmatizer.lemmatize(p, r.type.split(".")[-1]) for p in r.predictions]
            best_prediction = predictions[0] if predictions else ""
            top_substitutes = self.get_top_substitutes(r.substitutes)
            self.frame.at[r.Index, "best_score"] = self.get_vote_weight(best_prediction, r.substitutes)
            self.frame.at[r.Index, "best_mode_score"] = int(best_prediction in top_substitutes)
            self.frame.at[r.Index, "oot_score"] = sum([self.get_vote_weight(p, r.substitutes) for p in predictions])
            self.frame.at[r.Index, "oot_mode_score"] = int(any(p in top_substitutes for p in predictions))

    def get_average_scores(self):
        frame_predicted = self.frame[self.frame["predictions"].map(bool)]
        frame_non_tie = frame_predicted[frame_predicted["tie"] == False]
        return {
            "best_score": frame_predicted["best_score"].mean().item(),
            "best_mode_score": frame_non_tie["best_mode_score"].mean().item(),
            "oot_score": frame_predicted["oot_score"].mean().item(),
            "oot_mode_score": frame_non_tie["oot_mode_score"].mean().item()
        }

    def get_predictions(self, text_id, substituter, text, target, position):
        print(f"\rLoading predictions for instance id: {text_id:<5}", end='', flush=True)
        try:
            return list(substituter.substitute(text, target, position))[:10]
        except Exception as e:
            print(f"Error: {e}")
        return []

    def get_vote_weight(self, prediction, substitute_map):
        return substitute_map.get(prediction, 0) / sum(substitute_map.values())

    def get_top_substitutes(self, substitute_map):
        return [s for s, count in substitute_map.items() if count == max(substitute_map.values())]
