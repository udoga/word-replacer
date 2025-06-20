from nltk.stem import WordNetLemmatizer

class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def get_lemma(self, word, pos):
        return self.lemmatizer.lemmatize(word.lower(), pos)

    def has_same_root(self, a, b):
        for pos in ('v', 'n', 'a', 'r'):
            if self.get_lemma(a, pos) == self.get_lemma(b, pos):
                return True
        return False
