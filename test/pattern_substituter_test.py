from unittest import TestCase

from source.mock_substituter import MockSubstituter
from source.pattern_substituter import PatternSubstituter
from source.substitution_request import SubstitutionRequest

class PatternSubstituterTest(TestCase):
    def setUp(self):
        self.mock = MockSubstituter()
        self.substituter = PatternSubstituter(self.mock, ["%", "(", "means", "%", ")"], 3)

    def test_preprocesses_text_and_position(self):
        self.mock.load_responses([["strong"], ["he"]])
        response_one = self.substituter.substitute(SubstitutionRequest("wine is stronger to drink.", "strong", 2))
        response_two = self.substituter.substitute(SubstitutionRequest("he concerns while he is gone .", "he", 3))
        self.assertEqual("wine is stronger ( means stronger ) to drink.", self.mock.requests[0].text)
        self.assertEqual("he concerns while he ( means he ) is gone .", self.mock.requests[1].text)
        self.assertEqual(5, self.mock.requests[0].position)
        self.assertEqual(6, self.mock.requests[1].position)
        self.assertEqual(self.mock.responses, [response_one, response_two])
