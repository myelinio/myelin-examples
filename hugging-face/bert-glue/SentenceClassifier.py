import os
from transformers import (
	BertTokenizer,
	TFBertForSequenceClassification,
)

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class SentenceClassifier(object):

	def __init__(self):
		self.model = TFBertForSequenceClassification.from_pretrained(os.path.join(model_path, "save/"))
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

	def predict(self, X, feature_names):
		sentence1 = X[0]
		sentence2 = X[1]
		inputs = self.tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True, return_tensors='tf')
		pred = self.model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0].numpy().argmax().item()
		return [pred]
