### Github insights Spark job

This example runs a Spark job that writes results to BigQuery.

### Additional setup

Spark setup on Kubernetes

```bash
gcloud iam service-accounts create spark-bq --display-name spark-bq
export SA_EMAIL=$(gcloud iam service-accounts list --filter="displayName:spark-bq" --format='value(email)')
export PROJECT=$(gcloud info --format='value(config.project)')
gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA_EMAIL --role roles/storage.admin
gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA_EMAIL --role roles/bigquery.dataOwner
gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA_EMAIL --role roles/bigquery.jobUser
gcloud iam service-accounts keys create spark-sa.json --iam-account $SA_EMAIL
kubectl create secret generic spark-sa --from-file=spark-sa.json --namespace=myelin
kubectl create clusterrolebinding user-admin-binding --clusterrole=cluster-admin --user=$(gcloud config get-value account)
kubectl create clusterrolebinding --clusterrole=cluster-admin --serviceaccount=default:default spark-admin
kubectl create serviceaccount spark --namespace=myelin
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=myelin:spark --namespace=myelin
```

More info [here.](https://cloud.google.com/solutions/spark-on-kubernetes-engine)