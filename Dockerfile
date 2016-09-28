FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV GOOGLE_APPLICATION_CREDENTIALS ./image-annotator-key.json
ENTRYPOINT ["python"]
CMD ["image-annotator.py"]