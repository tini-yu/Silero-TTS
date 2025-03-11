FROM python:3.12

WORKDIR /silero

COPY ./requirements.txt ./requirements.txt

RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --upgrade --default-timeout=10000 -r ./requirements.txt

COPY . .

EXPOSE 8002

CMD ["python3", "-u", "./main.py", "--port", "8002"]
