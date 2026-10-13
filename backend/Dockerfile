FROM python:3.10-slim

# Without this, stdout/stderr are fully block-buffered once they're not a
# TTY (as in a container) -- output can sit unflushed and, on a hard kill
# (this project has had real OOM kills before), be lost outright rather
# than just delayed.
ENV PYTHONUNBUFFERED=1

# glibc's malloc keeps one arena per thread by default -- with Basic
# Pitch/torch's internal thread pools that fragments the heap across
# arenas, each with its own retained-but-unused pages _release_memory_to_os's
# malloc_trim can't reach (it only trims the main arena). Capping arenas
# caps that fragmentation on the 512MiB tier. OMP_NUM_THREADS=1 similarly
# caps OpenMP (used by numpy/scipy under the hood) from spawning one thread
# per core -- irrelevant to correctness on the free tier's 0.1 CPU, but
# each extra thread has its own stack + thread-local allocator state.
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1

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
