import pandas as pd
import csv

class BenchmarkReporter:
    def __init__(self, dataset_folder):
        self.frame = None
        self.dataset_folder = dataset_folder

    def load_dataset(self, file_prefix):
        self.load_sentences(file_prefix + ".preprocessed")
        self.load_substitutes(file_prefix + ".gold")

    def load_sentences(self, file_name):
        self.frame = pd.read_csv(self.dataset_folder + file_name, names=["target", "id", "position", "text"],
                                 sep="\t", header=None, encoding="iso-8859-1", engine="python", quoting=csv.QUOTE_NONE)
        self.frame = self.frame.set_index("id")
        self.frame[['target', 'type']] = (self.frame['target'].str.rsplit('.', n=1, expand=True))
        self.frame['substitutes'] = [{} for _ in range(len(self.frame))]

    def load_substitutes(self, file_name):
        with open(self.dataset_folder + file_name, 'r', encoding="iso-8859-1") as file:
            for line in file:
                left, right = [p.strip() for p in line.strip().split("::")]
                idx = int(left.split()[-1])
                substitutes = {" ".join(c.split()[:-1]): int(c.split()[-1]) for c in right.split(";") if c.strip()}
                self.frame.at[idx, "substitutes"] = substitutes

    def get_frame(self) -> pd.DataFrame:
        return self.frame

