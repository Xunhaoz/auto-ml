FROM python:3.10-slim AS install-base

RUN apt-get update -y
RUN apt-get install libgomp1

RUN pip install --upgrade pip
RUN pip install gunicorn

FROM install-base AS install-requirements

WORKDIR /app
COPY './requirements.txt' .
RUN pip --timeout=1000 install -r requirements.txt

FROM install-requirements AS release

WORKDIR /app
COPY . .

CMD ["gunicorn" , "-b", "0.0.0.0:9025", "app:app"]
EXPOSE 9025