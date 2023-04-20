FROM python:3.7

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD streamlit run app2.py