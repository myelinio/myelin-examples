FROM maven AS mavenbuild

#ENV BUCKET_PATH=gs://myelin-development-spark-on-k8s2/jars/

WORKDIR /app

ADD ./ /app

RUN mvn clean package

FROM google/cloud-sdk

#ENV GOOGLE_APPLICATION_CREDENTIALS /var/secrets/google/service_key.json/
COPY /var/secrets/google/ /var/secrets/google/

RUN ls /var/secrets/google/

WORKDIR /deploy
#COPY ./github-insights-1.0-SNAPSHOT-jar-with-dependencies.jar .
COPY --from=mavenbuild /app/target/github-insights-1.0-SNAPSHOT-jar-with-dependencies.jar .

RUN ls /deploy

RUN gsutil cp ./github-insights-1.0-SNAPSHOT-jar-with-dependencies.jar gs://myelin-development-spark-on-k8s2/jars/


FROM docker.io/myelinio/spark-client:v2.3.0