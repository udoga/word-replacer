import pandas as pd

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

    def __str__(self):
        frame = self.to_frame()
        frame.loc["Total"] = frame.sum(numeric_only=True)
        return frame.to_string()
