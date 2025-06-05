from dropout_substituter import DropoutSubstituter
from roberta_model import RobertaModel

model = RobertaModel()
substituter = DropoutSubstituter(model, 0.5, 10)
text = "The wine he sent to me as my birthday gift is too powerful to drink."
target = "powerful"
results = substituter.substitute(text, target)

print(results)
