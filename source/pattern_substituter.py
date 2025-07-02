class PatternSubstituter:
    def __init__(self, next_substituter, replacement_words, position_change):
        self.substituter = next_substituter
        self.replacement_words = replacement_words
        self.position_change = position_change

    def substitute(self, text, target, position):
        return self.substituter.substitute(self.preprocess(text, position), target, position + self.position_change)

    def preprocess(self, text, position):
        words = text.split()
        words[position:position + 1] = [w.replace("%", words[position]) for w in self.replacement_words]
        return " ".join(words)
