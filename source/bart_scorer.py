from external.bart_score import BARTScorer

class BartScorer:
    def __init__(self):
        self.name = "bart_score"
        self.scorer = BARTScorer(device="cpu", checkpoint='facebook/bart-large-cnn')

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
