FROM maven AS mavenbuild

#ENV BUCKET_PATH=gs://myelin-development-spark-on-k8s2/jars/
RUN mkdir -p /var/secrets/google/
ADD secrets/ /var/secrets/google/

WORKDIR /app

ADD ./ /app

RUN mvn clean package

FROM google/cloud-sdk

RUN mkdir -p /var/secrets/google/
ADD secrets/ /var/secrets/google/
RUN ls /var/secrets/google/

WORKDIR /deploy

#COPY ./github-insights-1.0-SNAPSHOT-jar-with-dependencies.jar .
COPY --from=mavenbuild /app/target/github-insights-1.0-SNAPSHOT-jar-with-dependencies.jar .


RUN ls /deploy
ENV GOOGLE_APPLICATION_CREDENTIALS /var/secrets/google/spark-sa.json
RUN gcloud config list

COPY gcp/ /root/.config/gcloud
RUN gcloud config list


RUN gcloud auth activate-service-account --key-file=/var/secrets/google/spark-sa.json

RUN gcloud config list

RUN gsutil cp ./github-insights-1.0-SNAPSHOT-jar-with-dependencies.jar gs://myelin-development-spark-on-k8s2/jars/


FROM docker.io/myelinio/spark-client:v2.3.0