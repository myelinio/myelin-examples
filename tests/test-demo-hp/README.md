# Test example

This example demonstrate how to use hyperparameter tuning. 

# Example predict

```
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE ml-test-hp -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray": [[1,2,3],[5,6,7]]}}'
curl -v -d "${DATA}" "${PROXY_URL}"
```