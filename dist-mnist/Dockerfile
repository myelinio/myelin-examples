FROM tensorflow/tensorflow:1.15.5-py3

COPY requirements.train.txt requirements.txt
RUN pip install -r  requirements.txt

ADD . /var/tf_dist_mnist
ENTRYPOINT ["python", "/var/tf_dist_mnist/dist_mnist.py"]