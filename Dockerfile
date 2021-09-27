FROM python:3

COPY  . src/

RUN pip freeze > /src/requirements.txt

RUN pip install -r  /src/requirements.txt

CMD [ "python", "/src/model_trainer.py" ]