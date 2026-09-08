FROM python:3.10-slim

WORKDIR /app
COPY . /app/

# fluidsynth: MIDI -> WAV synthesis. ffmpeg: decode fallback for browser
# recordings (webm/opus) and any container libsndfile can't read (mp3/m4a) --
# inference.py's _read_audio_bytes shells out to `ffmpeg` when librosa.load
# raises. Without it, every recorded-audio transcription fails with
# "FileNotFoundError: 'ffmpeg'" -> "transcription failed unexpectedly".
RUN apt-get update \
    && apt-get install -y --no-install-recommends fluidsynth ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
