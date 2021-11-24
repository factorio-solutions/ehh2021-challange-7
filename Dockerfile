FROM python:3.8.8-buster

LABEL version="2021.4.1" maintainer="petr.cezner@factorio.cz"

#RUN apt-get update -y && \
#    apt-get install -y python-pip python-dev

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1


COPY factorio/. /app/factorio
COPY mnt/. /app/mnt


EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["factorio/streamlit/main.py", "--", "-c", "factorio/config.ini"]