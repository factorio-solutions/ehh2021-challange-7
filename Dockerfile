FROM python:3.8.12-buster

LABEL version="2021.4.1" maintainer="petr.cezner@factorio.cz"

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.0

RUN pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"


WORKDIR /app

COPY poetry.lock /app/
COPY pyproject.toml /app/


COPY factorio/. /app/factorio
COPY mnt/ikem/. /app/mnt/ikem

COPY mnt/scaler.pkl /app/mnt/scaler.pkl
COPY mnt/model_state*.pth /app/mnt/

RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["factorio/streamlit/main.py", "--", "-c", "factorio/config.ini"]