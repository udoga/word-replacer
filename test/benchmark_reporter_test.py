from unittest import TestCase
import pandas as pd

from source.mock_substituter import MockSubstituter
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

    def test_loads_desired_number_of_rows(self):
        self.reporter.load_dataset("lst_trial", 1)
        df = self.reporter.get_frame()
        self.assertEqual(1, len(df))

    def test_calculates_best_scores(self):
        self.reporter.load_dataset("lst_trial", 3)
        self.substituter.load_responses([["intelligent"], ["luminous"], ["brilliant"]])
        self.reporter.load_scores(self.substituter)
        self.assertEqual([3/7, 2/5, 1/5], self.reporter.get_frame()["best_score"].to_list())

    def test_calculates_best_mode_scores(self):
        self.reporter.load_dataset("lst_trial", 3)
        self.substituter.load_responses([["clever"], ["luminous"], ["brilliant"]])
        self.reporter.load_scores(self.substituter)
        self.assertEqual([1, 0], self.reporter.get_frame()["best_mode_score"].to_list()[1:])

    def test_calculates_oot_scores(self):
        self.reporter.load_dataset("lst_trial", 3)
        self.substituter.load_responses([["x", "smart"], ["x", "x"], ["gleam", "colourful"]])
        self.reporter.load_scores(self.substituter)
        self.assertEqual([1/7, 0, 1/5 + 2/5], self.reporter.get_frame()["oot_score"].to_list())

    def test_calculates_oot_mode_scores(self):
        self.reporter.load_dataset("lst_trial", 3)
        self.substituter.load_responses([["x", "clever"], ["x", "luminous"], ["gleam", "x"]])
        self.reporter.load_scores(self.substituter)
        self.assertEqual([1, 0], self.reporter.get_frame()["oot_mode_score"].to_list()[1:])

    def test_gives_error_when_prediction_count_is_not_ten(self):
        self.reporter.load_dataset("lst_trial", 1)
        self.substituter.load_responses([["clever"]])
        self.substituter.responses[0].pop()
        self.assertRaises(AssertionError, self.reporter.load_scores, self.substituter)
