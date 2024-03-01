FROM python:3.8

RUN apt update -y && apt install awscli -y

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 9095

CMD ["python","app.py"]