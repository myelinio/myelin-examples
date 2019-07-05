#!/bin/bash
set -e

gcloud auth activate-service-account --key-file=/var/secrets/google/spark-sa.json spark-bq@myelin-development.iam.gserviceaccount.com
gsutil -cp ./github-insights-1.0-SNAPSHOT-jar-with-dependencies.jar gs://myelin-development-spark-on-k8s2/jars/

exec "$@"