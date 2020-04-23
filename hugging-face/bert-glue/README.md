# Hugging Face example

This mode is based on from [here.](https://github.com/huggingface/transformers/blob/master/examples/run_tf_glue.py)

# Example predict

```
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE bert-glue -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray": ["This research was consistent with his findings.", "His findings were compatible with this research."]}}'
curl -d "${DATA}" "${PROXY_URL}"
```

The API returns whether sentence one is a paraphrase of sentence two.