import pandas as pd
import pandas.testing as pd_testing
from unittest import TestCase
from source.substitution_table import SubstitutionTable

class SubstitutionTableTest(TestCase):
    def setUp(self):
        self.substitution_table = SubstitutionTable()

    def test_averages_values_in_frames(self):
        df1 = pd.DataFrame({'score1': [12, 56], 'score2': [34, 78]}, index=['abc', 'def'])
        df2 = pd.DataFrame({'score1': [10, 30], 'score2': [20, 40]}, index=['abc', 'xyz'])
        df3 = pd.DataFrame({'score1': [22, 56, 30], 'score2': [54, 78, 40]}, index=['abc', 'def', 'xyz'])
        for df in [df1, df2, df3]: df.index.name = 'candidate'
        sum_frame = SubstitutionTable.sum_frames([df1, df2])
        avg_frame = SubstitutionTable.avg_frames([df1, df2])
        pd_testing.assert_frame_equal(sum_frame, df3, rtol=1e-5, atol=1e-8)
        self.assertEqual(avg_frame.loc["abc"].tolist(), [11, 27])

    def test_dict_frame_conversions(self):
        table = SubstitutionTable({'candidate': ['a', 'b'], 'score': [1, 2]})
        frame = table.to_frame()
        new_table = SubstitutionTable.from_frame(frame)
        self.assertEqual(table, new_table)
