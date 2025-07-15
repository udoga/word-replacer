class MockSubstituter:
    def __init__(self):
        self.requests = []
        self.responses = []
        self.next_index = 0

    def substitute(self, text, target, position, tag):
        self.requests.append({"text": text, "target": target, "position": position, "tag": tag})
        if self.next_index >= len(self.responses): return []
        response = self.responses[self.next_index]
        self.next_index += 1
        return response

    def load_responses(self, responses):
        self.responses = responses
        self.next_index = 0
