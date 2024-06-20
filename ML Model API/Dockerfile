FROM python:3.10-slim

ENV PYTHONUNBUFFERED True
ENV PORT 8080

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "main.py"]
