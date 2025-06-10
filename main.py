from dropout_substituter import DropoutSubstituter
from roberta_model import RobertaModel

model = RobertaModel()
substituter = DropoutSubstituter(model, 0.5, 50)
text = "The wine he sent to me as my birthday gift is too powerful to drink"
target = "powerful"
table = substituter.substitute(text, target)
table.print_report()
