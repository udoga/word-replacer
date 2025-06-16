from unittest import TestCase
import pandas as pd

from .mock_substituter import MockSubstituter
from source.benchmark_reporter import BenchmarkReporter

class BenchmarkReporterTest(TestCase):
    def setUp(self):
        self.substituter = MockSubstituter()
        self.reporter = BenchmarkReporter("../dataset/")

    def test_loads_sentences(self):
        self.reporter.load_sentences("lst_test.preprocessed")
        df = self.reporter.get_frame()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual("side", df.loc[302]["target"])
        self.assertEqual("n", df.loc[302]["type"])
        self.assertEqual(2, df.loc[302]["position"])
        self.assertEqual("he told me , \" stay with your civilian clothes .", df.loc[317]["text"])
        self.assertEqual({}, df.loc[302]["substitutes"])
        self.assertEqual({}, df.loc[317]["substitutes"])

    def test_loads_substitutes(self):
        self.reporter.load_sentences("lst_test.preprocessed")
        self.reporter.load_substitutes("lst_test.gold")
        df = self.reporter.get_frame()
        self.assertEqual(1, df.loc[302]["substitutes"]["for us"])
        self.assertEqual({"say to": 3, "order": 1, "instruct": 1, "assure": 1}, df.loc[317]["substitutes"])

    def test_loads_sentences_and_substitutes(self):
        self.reporter.load_dataset("lst_all")
        df = self.reporter.get_frame()
        self.assertEqual("bright", df.loc[1]["target"])
        self.assertEqual(3, df.loc[1]["substitutes"]["intelligent"])
