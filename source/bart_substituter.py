from pandas import DataFrame
from source.substitution_table import SubstitutionTable
from source.substitution_request import SubstitutionRequest
from external.bart_score import BARTScorer

class BartSubstituter:
    def __init__(self, proposer=None):
        self.proposer = proposer
        self.scorer = BARTScorer(device="cpu", checkpoint='facebook/bart-large-cnn')

    def substitute(self, r: SubstitutionRequest) -> SubstitutionTable:
        frame = DataFrame({"candidate": r.candidates if r.candidates else []}).set_index("candidate")
        if self.proposer: frame = self.proposer.substitute(r).to_frame()
        frame["bart_score"] = self.score_candidates(r.text, r.position, frame.index.tolist())
        return SubstitutionTable.from_frame(frame.sort_values(by="bart_score", ascending=False))

    def score_candidates(self, text, position, candidates):
        words = text.split()
        candidate_sentences = []
        for w in candidates:
            candidate_sentences.append(self.get_candidate_sentence(words, position, w))
        scores = self.scorer.score([text] * len(candidate_sentences), candidate_sentences)
        return scores

    def get_candidate_sentence(self, original_words, position, candidate_word):
        words = original_words.copy()
        words[position] = candidate_word
        return ' '.join(words)
