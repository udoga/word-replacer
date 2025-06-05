from transformers import RobertaTokenizer, RobertaForMaskedLM
from dropout_substituter import DropoutSubstituter

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True)
substituter = DropoutSubstituter(model, tokenizer, 0.5, 10)

text = "The wine he sent to me as my birthday gift is too powerful to drink."
target = "powerful"
results = substituter.substitute(text, target)

print(results)
