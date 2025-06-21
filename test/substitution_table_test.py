from unittest import TestCase
from source.substitution_table import SubstitutionTable

class SubstitutionTableTest(TestCase):
    def setUp(self):
        self.substitution_table = SubstitutionTable()

    def test_dict_frame_conversions(self):
        table = SubstitutionTable({'candidate': ['a', 'b'], 'score': [1, 2]})
        frame = table.to_frame()
        new_table = SubstitutionTable.from_frame(frame)
        self.assertEqual(table, new_table)
