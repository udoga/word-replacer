from transformers import RobertaTokenizer, RobertaForMaskedLM
from source.dropout_substituter import DropoutSubstituter

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True, output_attentions=True, attn_implementation="eager")

substituter = DropoutSubstituter(tokenizer, model, dropout_rate=0.3, candidate_count=50, alpha=0.01, iteration_count=1, deterministic=True)
text = "The wine he sent to me as my birthday gift is too powerful to drink"
target = "powerful"
table = substituter.substitute(text, target)
table.print_report()
