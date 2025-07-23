from source.substitution_request import SubstitutionRequest

class MockSubstituter:
    def __init__(self):
        self.requests = []
        self.responses = []
        self.next_index = 0

    def substitute(self, r: SubstitutionRequest):
        self.requests.append(r)
        if self.next_index >= len(self.responses): return []
        response = self.responses[self.next_index]
        self.next_index += 1
        return response

    def load_responses(self, responses):
        self.responses = responses
        self.next_index = 0
