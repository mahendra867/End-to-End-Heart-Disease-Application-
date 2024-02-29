FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8081

CMD ["python","app.py"]