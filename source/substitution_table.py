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
        totals = frame.select_dtypes(include='number').sum()
        frame.loc['Total', totals.index] = totals
        return frame.to_string(float_format="%0.6f")

    def __iter__(self):
        frame = self.to_frame()
        filtered_frame = frame[frame['is_included']] if 'is_included' in frame else frame
        return iter(filtered_frame.index.values.tolist())
