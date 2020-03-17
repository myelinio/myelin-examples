# XGBoost trained on the MNIST dataset

This example trains and (and tunes) the model to classify handwritten digits.

# Example predict

```
NAMESPACE=myelin
URL=$(myelin endpoint -n $NAMESPACE xgboost-mnist -o json | jq -r '.[0].modelStable.publicUrl')
PROXY_URL=${URL}predict
DATA='{"data": {"ndarray":  [[ 0.0,  0.0,  1.0, 12.0,  5.0,  0.0,  0.0,  0.0,
                               0.0,  0.0,  9.0, 16.0, 14.0,  3.0,  0.0,  0.0,
                               0.0,  2.0, 16.0, 14.0, 11.0, 13.0,  0.0,  0.0,
                               0.0,  2.0, 16.0, 10.0,  0.0, 14.0,  4.0,  0.0,
                               0.0,  4.0, 16.0,  0.0,  0.0, 12.0,  4.0,  0.0,
                               0.0,  4.0, 16.0,  3.0,  0.0, 11.0, 10.0,  0.0,
                               0.0,  0.0, 13.0, 12.0,  8.0, 14.0,  6.0,  0.0,
                               0.0,  0.0,  3.0, 10.0, 16.0, 12.0,  1.0,  0.0]]}}'
curl -d "${DATA}" "${PROXY_URL}"
```

This sends a pixel representation for the digit and the model return a prediction of what number it might be.