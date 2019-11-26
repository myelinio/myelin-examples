FROM maven AS mavenbuild

RUN mkdir -p /var/secrets/google/
ADD secrets/ /var/secrets/google/

WORKDIR /app

ADD ./ /app

RUN mvn clean package

WORKDIR /deploy

COPY --from=mavenbuild /app/target/pi-example-1.0-SNAPSHOT-jar-with-dependencies.jar .

COPY copyArtifact.sh /copyArtifact.sh

RUN chmod +x /copyArtifact.sh
RUN sh -c /copyArtifact.sh

FROM docker.io/myelinio/spark-client:v2.3.0