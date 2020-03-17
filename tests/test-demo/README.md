# Test example

This is a simple example to train a scikit-learn model deploy it as two separate models with
weighed routing. 

# Example predict

```
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE ml-test -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray": [[1,2,3],[5,6,7]]}}'
curl -d "${DATA}" "${PROXY_URL}"
```