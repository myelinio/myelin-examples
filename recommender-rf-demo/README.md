# Random Forest Recommender Demo

This example trains a recommender on the MovieLens 1m dataset and deploys the train model behind a rest API.

# Example predict

```
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE rec-rf-demo -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray": [3, 4]}}'
curl -d "${DATA}" "${PROXY_URL}"
```

This returns a list of recommendation for user 3 and 4.