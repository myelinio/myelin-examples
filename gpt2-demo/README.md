# GPT2 demo

This example downloads a pre-trained GPT2 model and fine tunes is to the
complete works of Shakespeare. Note that it trains very slow on CPUs. More 
info about this example can be found [here](https://myelin.io/docs/examples/nlp/gpt2-demo/).  

# Example predict

```
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE gpt2-demo -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray": ["To be, or not to be: that is the question"]}}'
curl -v -d "${DATA}" "${PROXY_URL}"
```

The API returns some generated text based on this seed sentence.