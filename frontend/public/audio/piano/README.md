# Piano samples

Used by `components/midi-player.tsx` (via `hooks/use-midi-player.ts`) to
synthesize generated melodies in the browser with `Tone.Sampler`. Committed on
purpose so playback has no third-party CDN dependency — see `docs/BASELINE.md`.

**Source:** Salamander Grand Piano V3 by Alexander Holm, via the Tone.js sample
set at <https://github.com/tonejs/tonejs.github.io/tree/master/audio/salamander>.

**Licence:** Creative Commons Attribution 3.0
(<https://creativecommons.org/licenses/by/3.0/>).

Minimal subset — one sample every tritone from C2 to C6 (`Tone.Sampler`
pitch-shifts to cover the notes in between). ~0.65 MB total. Filenames use `s`
for sharp (`Fs4.mp3` = F♯4).
