from unittest import TestCase
from pathlib import Path
from statistics import mean
import pandas as pd

from source.mock_substituter import MockSubstituter
from source.benchmark_reporter import BenchmarkReporter

class BenchmarkReporterTest(TestCase):
    def setUp(self):
        self.substituter = MockSubstituter()
        self.reporter = BenchmarkReporter(Path(__file__).resolve().parents[1] / "dataset")

    def test_loads_sentences(self):
        self.reporter.load_sentences("lst_test.preprocessed")
        df = self.reporter.get_frame()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual("side", df.loc[302]["target"])
        self.assertEqual("n", df.loc[302]["tag"])
        self.assertEqual(2, df.loc[302]["position"])
        self.assertEqual("he told me , \" stay with your civilian clothes .", df.loc[317]["text"])
        self.assertEqual({}, df.loc[302]["gold_map"])
        self.assertEqual({}, df.loc[317]["gold_map"])

    def test_loads_gold_map(self):
        self.reporter.load_sentences("lst_test.preprocessed")
        self.reporter.load_gold_map("lst_test.gold")
        df = self.reporter.get_frame()
        self.assertEqual(1, df.loc[302]["gold_map"]["ally"])
        self.assertTrue("for us" not in df.loc[302]["gold_map"]) # removes multi-word golds
        self.assertEqual({"order": 1, "instruct": 1, "assure": 1}, df.loc[317]["gold_map"])

    def test_loads_sentences_and_substitutes(self):
        self.reporter.load_dataset("lst_all")
        df = self.reporter.get_frame()
        self.assertEqual("bright", df.loc[1]["target"])
        self.assertEqual(3, df.loc[1]["gold_map"]["intelligent"])

    def test_loads_desired_number_of_rows(self):
        self.reporter.load_dataset("lst_trial", row_count=1)
        df = self.reporter.get_frame()
        self.assertEqual(1, len(df))

    def test_loads_predictions(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["intelligent"], ["luminous"], ["brilliant"]])
        self.reporter.load_predictions(self.substituter)
        predictions_column = self.reporter.get_frame()["predictions"].to_list()
        self.assertEqual(["intelligent"], predictions_column[0])
        self.assertEqual(["luminous"], predictions_column[1])
        self.assertEqual(["brilliant"], predictions_column[2])

    def test_calculates_best_scores(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["intelligent"], ["luminous"], ["brilliant"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([3/7, 2/5, 1/5], self.reporter.get_frame()["best_score"].to_list())
        self.assertAlmostEqual(mean([3/7, 2/5, 1/5]), self.reporter.get_average_scores()["best_score"], places=6)

    def test_calculates_best_mode_scores(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["clever"], ["luminous"], ["brilliant"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([1, 1, 0], self.reporter.get_frame()["best_mode_score"].to_list())
        self.assertAlmostEqual(mean([1, 1, 0][1:]), self.reporter.get_average_scores()["best_mode_score"], places=6)

    def test_calculates_oot_scores(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["x", "smart"], ["x", "x"], ["gleam", "colourful"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([1/7, 0, 1/5 + 2/5], self.reporter.get_frame()["oot_score"].to_list())
        self.assertAlmostEqual(mean([1/7, 0, 1/5 + 2/5]), self.reporter.get_average_scores()["oot_score"], places=6)

    def test_calculates_oot_mode_scores(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["x", "clever"], ["x", "luminous"], ["gleam", "x"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([1, 1, 0], self.reporter.get_frame()["oot_mode_score"].to_list())
        self.assertAlmostEqual(mean([1, 1, 0][1:]), self.reporter.get_average_scores()["oot_mode_score"], places=6)

    def test_calculates_precision_1_scores(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["smart"], ["clear"], ["x", "colourful"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([1, 1, 0], self.reporter.get_frame()["precision@1"].to_list())
        self.assertAlmostEqual(mean([1, 1, 0]), self.reporter.get_average_scores()["precision@1"], places=6)

    def test_calculates_precision_3_scores(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["smart", "x", "clever"], ["x", "y", "clear", "light"], ["x", "colourful"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([2/3, 1/3, 1/3], self.reporter.get_frame()["precision@3"].to_list())
        self.assertAlmostEqual(mean([2/3, 1/3, 1/3]), self.reporter.get_average_scores()["precision@3"], places=6)

    def test_calculates_recall_10_scores(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["smart", "x", "clever"], ["x", "y", "clear", "light"], ["x", "colourful"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([2/3, 2/4, 1/4], self.reporter.get_frame()["recall@10"].to_list())
        self.assertAlmostEqual(mean([2/3, 2/4, 1/4]), self.reporter.get_average_scores()["recall@10"], places=6)

    def test_finds_if_there_is_tie_in_top_golds(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.assertEqual([True, False, False], self.reporter.get_frame()["tie"].to_list())

    def test_skips_instances_with_missing_predictions(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["intelligent"], [], ["colourful"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([3/7, 0.0, 2/5], self.reporter.get_frame()["best_score"].to_list())
        self.assertEqual([1, 0, 1], self.reporter.get_frame()["best_mode_score"].to_list())
        self.assertEqual([3/7, 0.0, 2/5], self.reporter.get_frame()["oot_score"].to_list())
        self.assertEqual([1, 0, 1], self.reporter.get_frame()["oot_mode_score"].to_list())
        self.assertAlmostEqual(mean([3/7, 2/5]), self.reporter.get_average_scores()["best_score"], places=6)
        self.assertAlmostEqual(mean([1]), self.reporter.get_average_scores()["best_mode_score"], places=6)
        self.assertAlmostEqual(mean([3/7, 2/5]), self.reporter.get_average_scores()["oot_score"], places=6)
        self.assertAlmostEqual(mean([1]), self.reporter.get_average_scores()["oot_mode_score"], places=6)

    def test_counts_prediction_as_correct_when_it_has_different_form(self):
        self.reporter.load_dataset("lst_trial", row_count=3)
        self.substituter.load_responses([["smartest"], ["clearer"], ["gleaming"]])
        self.reporter.load_predictions(self.substituter)
        self.reporter.load_scores()
        self.assertEqual([1/7, 1/5, 0], self.reporter.get_frame()["oot_score"].to_list())

    def test_excludes_rows_with_no_gold_data(self):
        self.reporter.load_dataset("lst_test")
        self.assertEqual("about", self.reporter.get_frame().loc[566].target)
        self.assertRaises(KeyError, lambda: self.reporter.get_frame().loc[567])
