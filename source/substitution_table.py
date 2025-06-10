import pandas as pd
import torch

class SubstitutionTable:
    def __init__(self):
        self.candidate_tokens = None
        self.candidate_probs = None
        self.normalized_probs = None
        self.proposal_scores = None
        self.target_similarities = None
        self.validation_scores = None
        self.final_scores = None

    def configure_display(self):
        torch.set_printoptions(linewidth=1000)
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', 500)

    def print_report(self):
        self.configure_display()
        print(self.create_frame())

    def create_frame(self):
        df = pd.DataFrame(data=dict(
            candidate_token=self.candidate_tokens,
            candidate_prob=self.candidate_probs,
            normalized_prob=self.normalized_probs,
            proposal_score=self.proposal_scores,
            target_similarity=self.target_similarities,
            validation_score=self.validation_scores,
            final_score=self.final_scores))
        df.loc["Total"] = df.sum(numeric_only=True)
        return df
