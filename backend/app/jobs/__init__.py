"""Asynchronous transcription job workflow.

/api/transcribe used to run Basic Pitch synchronously inside the HTTP
request, which tied up a request thread for the whole
GENERATION_TIMEOUT_SECONDS budget and couldn't be cancelled server-side once
the client gave up (see CLAUDE.md). That route is retired -- this package's
routes.py is now the only handler for /api/transcribe, backed by a
queue/worker boundary instead of running inline:

- routes.py     FastAPI endpoints: POST /transcribe, GET /transcribe/{id}
- service.py    create_transcription_job() + lazy store/queue/storage singletons
- store.py      JobStore: job metadata (SQLite dev, Postgres production)
- queue.py      JobQueue: work handoff (in-process dev, Redis production)
- storage.py    ObjectStorage: input audio bytes (local disk dev, S3/R2 production)
- worker.py     the actual claim -> run existing pipeline -> update status loop
"""
