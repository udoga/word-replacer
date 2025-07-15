import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

class CandidateExcluder:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def are_candidates_included(self, candidates, target="", tag=""):
        return [c.startswith(" ") and self.is_candidate_included(c.lower().strip(), target, tag) for c in candidates]

    def is_candidate_included(self, candidate, target, tag=""):
        return (not self.has_same_root(candidate, target)
                and not any(punctuation in candidate for punctuation in string.punctuation)
                and not candidate in stopwords.words("english")
                and (not tag or tag in self.get_possible_tags(candidate)))

    def has_same_root(self, a, b):
        for pos in ('v', 'n', 'a', 'r'):
            if self.lemmatizer.lemmatize(a.lower(), pos) == self.lemmatizer.lemmatize(b.lower(), pos):
                return True
        return False

    def get_possible_tags(self, word):
        return {('a' if p == 's' else p) for p in (syn.pos() for syn in wn.synsets(word))}
