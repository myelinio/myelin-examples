# Hugging Face example

This model is adapted from [this example.](https://github.com/huggingface/transformers/blob/master/examples/run_tf_glue.py)

### Example predict for task mrpc (Microsoft Research Paraphrase Corpus)

```bash
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE bert-glue -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray": ["This research was consistent with his findings.", "His findings were compatible with this research."]}}'
curl -d "${DATA}" "${PROXY_URL}"
```

The API returns whether sentence one is a paraphrase of sentence two.

### Example predict for task sst-2 (The Stanford Sentiment Treebank)

```bash
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE bert-glue -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
POSITIVE='{"data": {"ndarray": ["The consistently excellent Better Call Saul still has a little too much filler, but the series remains a worthy follow-up to Breaking Bad, and Magic Man points to this penultimate season setting the stage for a must-see final two years."]}}'
curl -d "${POSITIVE}" "${PROXY_URL}"
NEGATIVE='{"data": {"ndarray": ["For a show that regularly displays breathtaking technical mastery, and pairs it with so many compelling, laudable performances, there is an inescapable feeling of emptiness accompanying every shocking death or big plot reveal on Westworld."]}}'
curl -d "${NEGATIVE}" "${PROXY_URL}"
```