import os
from transformers import (
	BertTokenizer,
	TFBertForSequenceClassification,
)

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class SentenceClassifier():

	def __init__(self):
		self.model = TFBertForSequenceClassification.from_pretrained(os.path.join(model_path, "save/"))
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

	def predict(self, X, feature_names):
		s1 = X[0]
		s2 = X[1]
		inputs = self.tokenizer.encode_plus(s1, s2, add_special_tokens=True, return_tensors="pt")
		pred = self.model(**inputs)[0].argmax().item()
		return pred
