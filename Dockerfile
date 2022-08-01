FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update
RUN apt-get install python3-opencv -y

RUN mkdir -p /workspace

COPY ./ /workspace/coreset
RUN chmod 777 /workspace/coreset/*

WORKDIR /workspace/coreset
RUN pip install -r requirements.txt

RUN chmod 755 /workspace/coreset/src/*.py
ENV PYTHONUNBUFFERED=1
CMD ["python", "./src/main.py"]