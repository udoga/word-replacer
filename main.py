from source.dropout_substituter import DropoutSubstituter
from source.roberta_model import RobertaModel

model = RobertaModel()
substituter = DropoutSubstituter(model, dropout_rate=0.3, candidate_count=50, alpha=0.01, iteration_count=1, deterministic=True)
text = "The wine he sent to me as my birthday gift is too powerful to drink"
target = "powerful"
table = substituter.substitute(text, target)
table.print_report()
