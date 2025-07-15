from unittest import TestCase

from source.mock_substituter import MockSubstituter
from source.pattern_substituter import PatternSubstituter

class PatternSubstituterTest(TestCase):
    def setUp(self):
        self.mock = MockSubstituter()
        self.substituter = PatternSubstituter(self.mock, ["%", "(", "means", "%", ")"], 3)

    def test_preprocesses_text_and_position(self):
        self.mock.load_responses([["strong"], ["he"]])
        response_one = self.substituter.substitute("the wine he sent is stronger to drink.", "strong", 5)
        response_two = self.substituter.substitute("he concerns while he is gone .", "he", 3)
        self.assertEqual("the wine he sent is stronger ( means stronger ) to drink.", self.mock.requests[0]["text"])
        self.assertEqual("he concerns while he ( means he ) is gone .", self.mock.requests[1]["text"])
        self.assertEqual(8, self.mock.requests[0]["position"])
        self.assertEqual(6, self.mock.requests[1]["position"])
        self.assertEqual(self.mock.responses, [response_one, response_two])
