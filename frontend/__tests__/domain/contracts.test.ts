import { clipAnalysisSchema } from "../../src/domain/contracts/analysis";
import {
  createTranscribeJobResponseSchema,
  jobResultStatusSchema,
  transcribeJobStatusSchema,
} from "../../src/domain/contracts/job";
import {
  progressionJobResultSchema,
  progressionsResponseSchema,
  renderProgressionResponseSchema,
} from "../../src/domain/contracts/progressions";
import { uploadResponseSchema } from "../../src/domain/contracts/upload";
import { variantsResponseSchema } from "../../src/domain/contracts/variants";

// Payloads shaped like the FastAPI routes send them (backend/app/main.py,
// backend/app/jobs/routes.py, backend/app/inference.py).

const TRANSCRIPTION = {
  n_notes: 1,
  duration_sec: 2,
  source_duration_sec: 2,
  truncated: false,
  midi_b64: "AAA=",
  wav_b64: null,
  midi_filename: "transcription_x.mid",
  wav_filename: "",
  note_events: [{ start: 0, end: 0.5, pitch: 60, velocity: 90 }],
  transcribed_by: "server",
  n_chunks: 1,
};

const VARIANTS = {
  n_variants: 1,
  temperatures: [0.7],
  mood_idx: 0,
  mood_label: "happy",
  model_status: {
    cvae: { path: "cvae", exists: true, size_mb: 1, loaded: true },
    iddm_ppo: { path: "iddm", exists: true, size_mb: 1, loaded: true },
    device: "cpu",
    load_error: null,
    fluidsynth_available: true,
  },
  variants: [
    {
      index: 0,
      temperature: 0.7,
      midi_b64: "AAA=",
      midi_filename: "variant_1.mid",
      midi_download_path: "/api/download/variant_1.mid",
      wav_b64: null,
      wav_filename: "",
      wav_download_path: "",
    },
  ],
};

const PROGRESSION = {
  progression: ["C", "G"],
  bpm: 120,
  instrument: 0,
  audio_b64: "AAA=",
  wav_b64: "AAA=",
  filename: "progression.wav",
  download_path: "/api/download/progression.wav",
  midi_b64: "AAA=",
  midi_filename: "progression.mid",
  midi_download_path: "/api/download/progression.mid",
  wav_filename: "progression.wav",
  wav_download_path: "/api/download/progression.wav",
  detected_chords: ["C", "G"],
};

function status(result: unknown, overrides: Record<string, unknown> = {}) {
  return {
    job_id: "job-1",
    status: "completed",
    progress: 100,
    result,
    error: null,
    waiting_reason: null,
    ...overrides,
  };
}

describe("upload contract", () => {
  it("accepts POST /api/upload's 201 body", () => {
    expect(uploadResponseSchema.parse({ id: "abc", filename: "upload_abc.wav" })).toEqual({
      id: "abc",
      filename: "upload_abc.wav",
    });
  });

  it("rejects a body without an id", () => {
    expect(uploadResponseSchema.safeParse({ filename: "upload_abc.wav" }).success).toBe(false);
  });
});

describe("job contracts", () => {
  it("accepts POST /api/transcribe's 202 body, which carries only id and status", () => {
    expect(createTranscribeJobResponseSchema.parse({ job_id: "j", status: "queued" })).toEqual({
      job_id: "j",
      status: "queued",
    });
  });

  it("parses a completed transcription status", () => {
    const parsed = transcribeJobStatusSchema.parse(status(TRANSCRIPTION));
    expect(parsed.result?.n_notes).toBe(1);
  });

  it("accepts a job persisted before waiting_reason / transcribed_by / n_chunks existed", () => {
    const { transcribed_by: _by, n_chunks: _chunks, ...legacyResult } = TRANSCRIPTION;
    const { waiting_reason: _reason, ...legacyStatus } = status(legacyResult);
    expect(transcribeJobStatusSchema.safeParse(legacyStatus).success).toBe(true);
  });

  it("accepts the expired payload GET /transcribe/{id} sends past retention", () => {
    const expired = { job_id: "j", status: "expired", progress: 100, result: null, error: null };
    expect(jobResultStatusSchema.parse(expired).status).toBe("expired");
  });

  it("allows unknown extra keys", () => {
    expect(
      transcribeJobStatusSchema.safeParse({ ...status(TRANSCRIPTION), something_new: 1 }).success,
    ).toBe(true);
  });

  it("rejects an unknown job status", () => {
    expect(transcribeJobStatusSchema.safeParse(status(null, { status: "done" })).success).toBe(
      false,
    );
  });

  it("rejects a transcription result missing a field the UI reads", () => {
    const { n_notes: _n, ...broken } = TRANSCRIPTION;
    expect(transcribeJobStatusSchema.safeParse(status(broken)).success).toBe(false);
  });

  describe("job result union", () => {
    it("tags a variants result", () => {
      const parsed = jobResultStatusSchema.parse(status({ ...VARIANTS, job_id: "job-1" }));
      expect(parsed.result?.kind).toBe("variants");
    });

    it("tags a progression result", () => {
      const parsed = jobResultStatusSchema.parse(status(PROGRESSION));
      expect(parsed.result?.kind).toBe("progression");
    });

    it("tags a transcription result", () => {
      const parsed = jobResultStatusSchema.parse(status(TRANSCRIPTION));
      expect(parsed.result?.kind).toBe("transcription");
    });

    it("rejects a result that is none of the three", () => {
      expect(jobResultStatusSchema.safeParse(status({ hello: "world" })).success).toBe(false);
    });
  });
});

describe("analysis contract", () => {
  const ANALYSIS = {
    tempo_bpm: 120,
    average_pitch: 61.2,
    mood_idx: 0,
    mood_label: "happy",
    key: "C major",
    pitch_histogram: new Array(12).fill(0),
    detected_chords: ["C"],
    n_notes: 3,
    clip_start_sec: 0,
    clip_end_sec: 2,
  };

  it("accepts POST /api/analyze's body", () => {
    expect(clipAnalysisSchema.parse(ANALYSIS).key).toBe("C major");
  });

  it("rejects a mood the UI has no badge for", () => {
    expect(clipAnalysisSchema.safeParse({ ...ANALYSIS, mood_label: "angry" }).success).toBe(false);
  });
});

describe("variants contract", () => {
  it("accepts POST /api/generate-variants' body", () => {
    expect(variantsResponseSchema.parse({ ...VARIANTS, job_id: "job-v" }).job_id).toBe("job-v");
  });

  it("requires the job id the result link is built from", () => {
    expect(variantsResponseSchema.safeParse(VARIANTS).success).toBe(false);
  });

  it("rejects variants that aren't an array", () => {
    expect(
      variantsResponseSchema.safeParse({ ...VARIANTS, job_id: "j", variants: null }).success,
    ).toBe(false);
  });
});

describe("progressions contracts", () => {
  it("accepts GET /api/progressions' body", () => {
    const parsed = progressionsResponseSchema.parse({
      progressions: [
        {
          id: "pop-1",
          name: "I–V–vi–IV",
          genre: "Pop",
          source: "musicca",
          chords: ["C", "G", "Am", "F"],
          song_title: null,
          artist: null,
          year: null,
        },
      ],
    });
    expect(parsed.progressions).toHaveLength(1);
  });

  it("rejects a list item without chords", () => {
    expect(
      progressionsResponseSchema.safeParse({
        progressions: [{ id: "x", name: "x", genre: "Pop", source: "s" }],
      }).success,
    ).toBe(false);
  });

  it("accepts POST /api/generate-progression's body", () => {
    expect(renderProgressionResponseSchema.parse({ ...PROGRESSION, job_id: "job-p" }).job_id).toBe(
      "job-p",
    );
  });

  it("accepts a stored progression job result", () => {
    expect(progressionJobResultSchema.parse(PROGRESSION).bpm).toBe(120);
  });
});
