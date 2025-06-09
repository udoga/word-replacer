from config import configure_display
from dropout_substituter import DropoutSubstituter
from roberta_model import RobertaModel

configure_display()

model = RobertaModel()
substituter = DropoutSubstituter(model, 0.5, 30)
text = "The wine he sent to me as my birthday gift is too strong to drink"
target = "strong"
results = substituter.substitute(text, target)
results.loc["Total"] = results.sum(numeric_only=True)

print(results)
