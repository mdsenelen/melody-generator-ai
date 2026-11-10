// Audio files for upload tests. Size is faked rather than allocated, so a
// "60 MB" file costs nothing.

type AudioFileOptions = {
  name?: string;
  type?: string;
  /** Reported File.size; defaults to the real byte length of the contents. */
  size?: number;
  contents?: string;
};

export function audioFile({
  name = "riff.wav",
  type = "audio/wav",
  size,
  contents = "RIFF....WAVEfmt ",
}: AudioFileOptions = {}): File {
  const file = new File(size === 0 ? [] : [contents], name, { type });
  if (size !== undefined && size !== 0) {
    Object.defineProperty(file, "size", { value: size });
  }
  return file;
}

export function emptyFile(name = "silence.wav", type = "audio/wav"): File {
  return audioFile({ name, type, size: 0 });
}
