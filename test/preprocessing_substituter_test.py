from unittest import TestCase

from mock_substituter import MockSubstituter
from source.preprocessing_substituter import PreprocessingSubstituter

class PreprocessingSubstituterTest(TestCase):
    def setUp(self):
        self.substituter = MockSubstituter()
        self.preprocessor = PreprocessingSubstituter(self.substituter, ["%", "(", "means", "%", ")"], 3)

    def test_preprocesses_text_and_position(self):
        self.substituter.load_responses([["strong"], ["he"]])
        response_one = self.preprocessor.substitute("the wine he sent is stronger to drink.", "strong", 5)
        response_two = self.preprocessor.substitute("he concerns while he is gone .", "he", 3)
        self.assertEqual("the wine he sent is stronger ( means stronger ) to drink.", self.substituter.requests[0]["text"])
        self.assertEqual("he concerns while he ( means he ) is gone .", self.substituter.requests[1]["text"])
        self.assertEqual(8, self.substituter.requests[0]["position"])
        self.assertEqual(6, self.substituter.requests[1]["position"])
        self.assertEqual(self.substituter.responses, [response_one, response_two])
