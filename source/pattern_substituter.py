from source.substitution_request import SubstitutionRequest

class PatternSubstituter:
    def __init__(self, next_substituter, replacement_words, position_change):
        self.substituter = next_substituter
        self.replacement_words = replacement_words
        self.position_change = position_change

    def substitute(self, r: SubstitutionRequest):
        r.text = self.preprocess(r.text, r.position)
        r.position += self.position_change
        return self.substituter.substitute(r)

    def preprocess(self, text, position):
        words = text.split()
        words[position:position + 1] = [w.replace("%", words[position]) for w in self.replacement_words]
        return " ".join(words)
