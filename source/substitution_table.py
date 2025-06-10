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
        print(self.to_frame())

    def to_frame(self):
        df = pd.DataFrame(data=dict(
            candidate_token=self.candidate_tokens,
            candidate_prob=self.candidate_probs,
            normalized_prob=self.normalized_probs,
            proposal_score=self.proposal_scores,
            target_similarity=self.target_similarities,
            validation_score=self.validation_scores,
            final_score=self.final_scores))
        df = df.set_index('candidate_token')
        df.loc["Total:"] = df.sum(numeric_only=True)
        return df

    @staticmethod
    def avg_tables(substitution_tables):
        return SubstitutionTable.avg_frames_by_token([table.to_frame() for table in substitution_tables])

    @staticmethod
    def sum_frames_by_token(frames):
        return pd.concat([df.reset_index() for df in frames]).groupby(frames[0].index.name).sum()

    @staticmethod
    def avg_frames_by_token(frames):
        return SubstitutionTable.sum_frames_by_token(frames).apply(lambda x: x / len(frames))
