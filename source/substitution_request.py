class SubstitutionRequest:
    def __init__(self, text, target, position, tag=""):
        self.text = text
        self.target = target
        self.position = position
        self.tag = tag
