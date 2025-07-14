import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class CandidateExcluder:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def are_candidates_included(self, candidates, target=""):
        return [c.startswith(" ") and self.is_candidate_included(c.lower().strip(), target) for c in candidates]

    def is_candidate_included(self, candidate, target):
        return (not self.has_same_root(candidate, target)
                and not any(punctuation in candidate for punctuation in string.punctuation)
                and not candidate in stopwords.words("english"))

    def has_same_root(self, a, b):
        for pos in ('v', 'n', 'a', 'r'):
            if self.lemmatizer.lemmatize(a.lower(), pos) == self.lemmatizer.lemmatize(b.lower(), pos):
                return True
        return False
