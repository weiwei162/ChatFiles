FROM quay.io/unstructured-io/unstructured:latest

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD [ "python3.8", "server.py"]