import pandas as pd
import torch

class SubstitutionTable(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_frame(cls, frame):
        return cls(frame.reset_index().to_dict('list'))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self)
        df = df.set_index('candidate')
        return df

    def print_report(self):
        self.configure_display()
        frame = self.to_frame()
        frame.loc["Total"] = frame.sum(numeric_only=True)
        print(frame)

    @staticmethod
    def configure_display():
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', 500)

    @staticmethod
    def avg_tables(substitution_tables, order_by, size):
        frames = [table.to_frame() for table in substitution_tables]
        avg_frame = SubstitutionTable.avg_frames(frames).sort_values(by=[order_by], ascending=False).head(size)
        return SubstitutionTable.from_frame(avg_frame)

    @staticmethod
    def avg_frames(frames):
        return SubstitutionTable.sum_frames(frames).apply(lambda x: x / len(frames))

    @staticmethod
    def sum_frames(frames):
        return pd.concat([df.reset_index() for df in frames]).groupby(frames[0].index.name).sum()
