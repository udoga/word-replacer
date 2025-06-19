class MockSubstituter:
    def __init__(self):
        self.responses = []
        self.next_index = 0

    def get_predictions(self, text, target, target_index):
        if self.next_index >= len(self.responses): return []
        response = self.responses[self.next_index]
        self.next_index += 1
        return response

    def load_responses(self, responses):
        self.responses = responses
        for response in self.responses:
            response += ["x" for _ in range(10 - len(response))]
        self.next_index = 0
