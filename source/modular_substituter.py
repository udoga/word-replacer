from substitution_table import SubstitutionTable

class ModularSubstituter:
    def __init__(self, proposer, scorer):
        self.proposer = proposer
        self.scorer = scorer

    def substitute(self, text, target, position, tag="") -> SubstitutionTable:
        table = self.proposer.substitute(text, target, position, tag)
        table[self.scorer.name] = self.scorer.score_candidates(text, position, table['candidate'])
        return SubstitutionTable.from_frame(table.to_frame().sort_values(by=self.scorer.name, ascending=False))
