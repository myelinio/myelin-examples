# docker build -t myelinio/hpo-example-worker:v0.1.6 -f Dockerfile .
# docker run --network myelin-trainer  -e TRAIN_CONTROLLER_URL=http://a00240db09ac:5000 myelinio/hpo-example-worker:v0.1.0  --budget "6.172839506172839"
FROM python:3.6

RUN pip3 install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN for req in $(cat requirements.txt); do pip install $req; done

COPY . /app


ENTRYPOINT ["python", "xgboost_mnist.py"]