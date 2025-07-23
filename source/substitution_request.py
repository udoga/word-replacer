class SubstitutionRequest:
    def __init__(self, text, target, position, tag="", candidates=None):
        self.text = text
        self.target = target
        self.position = position
        self.tag = tag
        self.candidates = candidates
