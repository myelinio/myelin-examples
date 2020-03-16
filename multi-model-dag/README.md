# Multi model DAG deployment

This example shows how to create a a multi model deployment where models get executed in a DAG. In `axon-model-graph.yaml`
three models get deployed where model3 is dependent on model1 and model2.

# Example predict

```
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE multi-model-graph -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray": [[3, 4, 100], [4, 3, 1]]}}'
curl -v -d "${DATA}" "${PROXY_URL}"
``` 