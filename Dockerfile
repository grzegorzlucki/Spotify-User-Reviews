FROM python:3.11

WORKDIR /code/dir

COPY ./requirements.txt  /code/app/requirements.txt

RUN pip install --no-cache-dir -r /code/app/requirements.txt

COPY . /code/dir

CMD ["fastapi", "run", "main.py"]