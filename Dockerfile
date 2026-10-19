# Build the web app, then serve it and the API from one small Python image.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npx -y npm@11.6.2 ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
# LightGBM needs OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY snare/ snare/
COPY models/ models/
COPY --from=web /web/dist web/dist
# Simulate the month of history the live network starts from, once, here,
# so the server starts in seconds even on a small free instance.
RUN python -m snare.snapshot
# Hosts such as Render set PORT; 8000 otherwise.
ENV PORT=8000
EXPOSE 8000
RUN useradd --uid 1000 --create-home app
USER app
CMD ["sh", "-c", "exec uvicorn snare.server:app --host 0.0.0.0 --port ${PORT}"]
