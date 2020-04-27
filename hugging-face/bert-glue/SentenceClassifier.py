import os
from transformers import (
	BertTokenizer,
	TFBertForSequenceClassification,
)

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
task_type = os.environ.get('TASK_TYPE')


class SentenceClassifier(object):

	def __init__(self):
		self.model = TFBertForSequenceClassification.from_pretrained(os.path.join(model_path, "save/"))
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

	def predict(self, X, feature_names):
		# paraphrasing
		if task_type == "mrpc":
			if len(X) != 2:
				raise Exception("This task needs a pair of sentences.")
			inputs = self.tokenizer.encode_plus(X[0], X[1], add_special_tokens=True, return_tensors='tf')
			pred = self.model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0].numpy().argmax().item()
			return [pred]
		# semantic textual similarity
		elif task_type == "sts-b":
			if len(X) != 2:
				raise Exception("This task needs a pair of sentences.")
			inputs = self.tokenizer.encode_plus(X[0], X[1], add_special_tokens=True, return_tensors='tf')
			pred = self.model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0].numpy().item()
			return [pred]
		# sentiment
		elif task_type == "sst-2":
			if len(X) != 1:
				raise Exception("Only sentence is needed for this task.")
			inputs = self.tokenizer.encode_plus(X[0], add_special_tokens=True, return_tensors='tf')
			pred = self.model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0].numpy().argmax().item()
			return [pred]
