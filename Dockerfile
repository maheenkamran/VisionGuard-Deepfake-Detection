FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt
RUN useradd -m appuser
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser wsgi.py .
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "1", "wsgi:app"]