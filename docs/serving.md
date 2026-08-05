# Serving over HTTP

`tts-serve` replaces the Python FastAPI service on the same port, speaking the same
protocol, so a client switches by pointing at a different process.

```sh
# The Python service must not be holding the port.
TTS_API_KEY=… cargo run -p tts-serve --release -- --port 3003
```

`PORT` works too, as it does in the Python service's `run.sh`. Defaults: `127.0.0.1:3003`,
engine `cosyvoice`, voice `voices/cosy-default-cosyvoice`.

## What it serves

| route | status |
|---|---|
| `POST /tts` | **yes** — WAV body, same headers |
| `POST /tts/stream` | **yes**, but buffered — see below |
| `GET /health` | **yes** |
| `GET /v1/capabilities` | **yes** |
| `GET /` | **yes** — lists live and unimplemented routes |
| `POST /v1/tts-jobs`, `GET /v1/tts-jobs/{id}` | `501` |
| `POST /v1/alignment-jobs`, `GET /v1/alignment-jobs/{id}` | `501` |
| `GET /v1/artifacts/{job}/{file}` | `501` |

Anything unimplemented answers **501 with an explanation**, not 404 — the route is real,
this server just does not serve it. `GET /` enumerates both lists.

## Compatibility, verified

Both services were run side by side (Python on 3003, Rust on 3013) against the same key
and the same request:

- **Identical WAV format** — `RIFF`, `fmt=1` (PCM), mono, 24000 Hz, 16-bit.
- **Identical response headers** — `X-Audio-Seconds`, `X-Wall-Seconds`, `X-RTF`,
  `X-Audio-Format: pcm_s16le_mono`, `Content-Disposition`.
- **Identical auth** — `Authorization: Bearer …` or `X-API-Key: …`, constant-time compare,
  `503` when no key is configured, and the same `TTS_API_KEY` → `.api_key` fallback, so an
  existing deployment's secret does not move.
- **Errors** serialise as `{"detail": "…"}`, the shape FastAPI's `HTTPException` produces.

The audio differs between them, as it must: they sample from different RNG streams.

## Speed, honestly

Over HTTP, on `examples/senior.txt` (132 words), warm:

| | RTF | wall |
|---|---|---|
| Python service (MPS) | 0.80 | 39.8 s |
| `tts-serve` | **0.70** | 38.3 s |

**1.14×.** That is the number that matters for replacing the service, and it is far more
modest than the "6.27× vs stock PyTorch" figure elsewhere in these docs — that one compares
against upstream CosyVoice with no MPS path at all, which is the right baseline for *porting*
and the wrong one for *deployment*. Both are true; this is the one to quote when deciding
whether to switch.

Two other differences are worth more than the RTF:

- **Load time: 3.0–4.0 s against 15–17 s.** The Python service loads in a subprocess and
  reloads whenever its recycle budget trips.
- **No recycle budget.** `run.sh` sets `TTS_WORKER_MAX_GROWTH_MB`,
  `MAX_FOOTPRINT_MB`, `MAX_REQUESTS` and `IDLE_SECONDS` because, in its own words,
  *"PyTorch's MPS backend never frees its compiled-graph cache, so ending the process is
  the only way to reclaim it."* There is no such cache here, so the model stays resident
  and there is nothing to recycle.

Short requests do not show any of this: one sentence measures RTF ~1.0 to 1.2 because
Metal shader compilation has nothing to amortise over. Judge it on a paragraph.

## Knobs it refuses

`mode=instruct`, `mode=cross_lingual`, `speed != 1.0` and `instruct_text` all return
**501**. The port has no instruction-prompt path and synthesizes at 1.0 only.

Refusing rather than ignoring is deliberate, and it is the rule `tts_core::Sampling`
already states: an engine documents the controls it ignores instead of silently accepting
them. Returning speed-1.0 audio to a client that asked for 1.5 would report the request as
honoured when it was not, and nothing in the response would say otherwise.

Two additions the Python schema does not have, both optional so an existing client is
unaffected:

| field | |
|---|---|
| `voice` | a voice asset directory, per request, instead of the one the server started with |
| `seed` | make a request reproducible |

There is also an extra `X-Stages` response header carrying the per-stage split
(`llm=10.296,flow=25.129,vocoder=2.898`), so a client can see where the time went without
a second request.

## Concurrency

One GPU, so synthesis is serialised behind a semaphore: two requests interleaving on one
Metal queue make both slower and neither faster. Requests queue, and synthesis runs on
`spawn_blocking` so it never occupies an async worker. Throughput is one stream; the win
is that there is no per-request model load.

## `/tts/stream` is buffered

The bytes are correct — raw little-endian 16-bit mono PCM with `X-Sample-Rate`, exactly
what the Python route emits — but they are produced *after* full synthesis, not during it.
Neither engine exposes an incremental decode yet (`Capabilities::streaming` is `false` for
both), so the point of the Python route, time-to-first-audio, is not delivered. A client
needs no change and will not break; it will not get lower latency either. The response
carries `X-Streaming: buffered` so this is visible rather than implied.

Real streaming needs the flow decoder to run on chunks with a carried cache; see
[porting/cosyvoice.md](porting/cosyvoice.md).


## Narrating a long document

`scripts/narrate-book.sh` is the batch path: markdown in, WebM/Opus plus alignment manifests
out. One command from source to publishable:

```sh
scripts/narrate-book.sh --book path/to/document --out narration --engine cosyvoice
```

It loads the engine **once** for the whole run, refuses to start when another `tts` process
holds the GPU, and keeps going when one section fails rather than abandoning the rest.

Sections are discovered in either layout:

| layout | files |
|---|---|
| flat | `introduction.md`, `chapter-N.md` (numeric order), `conclusion.md` |
| nested | `part-NN-*/chapter-NNN-slug.md`, recursively, ordered by chapter number |

`_index.md` is skipped at every level — a landing page is not narration content. Output names
normalise to `chapter-NNN` so the published directory layout does not depend on the source
filename's slug.

Per section:

| step | tool | out |
|---|---|---|
| markdown -> narration text + page-word map | `md-to-narration.py --emit-map` | `.txt`, `.map.json` |
| text -> audio | `tts-serve`, one server for the whole run | `.wav` master |
| audio -> delivery format | ffmpeg, WebM/Opus 48 kbps | `.webm` |
| audio + text -> timings | `align-narration.py` | `.manifest.json` |

`scripts/narrate.sh` is the older single-stage entry point and is kept only for one-off
renders; it has none of the resumability or verification below.

### Resume per stage, not per section

Synthesis is the only expensive step, so it is the only one that must never repeat:

- a section with a manifest is skipped entirely
- a section with a WAV master is not re-synthesised, only re-encoded and re-aligned
- the server starts only if something actually needs synthesising

A run killed three sections in therefore costs nothing to restart. Renders are deterministic
(seed 1234 by default), so a resumed run produces the same audio rather than a different
valid draw.

That determinism has a consequence worth stating plainly: **re-rendering unchanged text
cannot repair a bad draw.** It reproduces the same audio byte for byte. Use `--seed` to get a
different trajectory when a section comes out wrong for sampling reasons rather than input
reasons.

One guard exists because the alternative fails silently: when a WAV master already exists and
the converter now produces different text, the text is **not** overwritten and a warning is
printed. Otherwise deleting a manifest to force a re-align would align yesterday's audio
against today's text, producing a manifest that describes words the voice never said.

### Segmentation must be bounded

Text is split into paragraphs, then into segments of whole sentences within `max_chars`
(default 220). Segmentation decides prompt length, batch shape, where silence is inserted,
and where the waveform can be cut.

For a long time the budget was applied only when *merging* sentences, so a single sentence
longer than the budget passed through whole — one reached 566 characters, about 38 seconds of
speech. The AR loop masks `eos` only for the first `2 * text_tokens` steps and may end the
sequence anywhere after that. On those long segments it did, mid-clause.

Nothing downstream could detect it. The vocoder renders whatever tokens it receives, so the
output duration matches the tokens produced rather than the text that should have produced
them: a file that is internally consistent, correct in length, and missing a clause. **24
segments across one book, roughly two minutes of text present in the source and absent from
the audio.**

`tts_core::text::segment` now splits an over-long sentence at clause punctuation, then at word
boundaries, so no segment can exceed the budget. Clause punctuation is preferred because the
inserted segment gap lands where the voice would already pause. Tests pin the invariant.

The engine also compares each segment's speech tokens per character against the median for
the request, regenerates outliers, and fails with the offending text if they persist. A
median measured from this voice and this text is a better reference than a words-per-minute
constant. Its limit: a segment losing a 49-character clause out of 220 sits near 0.78 of the
median, inside any threshold loose enough to avoid false positives. Token counts cannot see a
partial loss inside a merged segment, which is why recognition is the acceptance test.

### Alignment is derived from the audio

Knowing the script suggests an attractive shortcut: skip recognition, cut the script into
sentence windows, give each a slice of the timeline proportional to its character count, and
let a forced aligner place known words inside each window. It is about five times cheaper.

Measured against the audio it produced a **median error of 4.8 s per word**, with only 11.9%
of words within half a second of where they were spoken. A forced aligner places words
*inside the span it is handed*, cannot reject a bad span, and cannot report receiving one.
Character-proportional spans assume a constant speaking rate, which narration violates at
every heading, paragraph break and inserted gap.

The statistics that shipped alongside it could not see the problem. `alignedShare` says words
*received* timestamps; a coverage figure says they *span the file*. Neither says a timestamp
is where the word is.

What `align-narration.py` does instead:

1. **Recognise** with batched `faster-whisper` and `word_timestamps=True` — measured **20.4x
   realtime** on CPU, 51 s for a 17-minute section. Batching matters; the unbatched path is
   7.8x on the same machine.
2. **Match** the recognised sequence against the document's canonical words with
   `difflib.SequenceMatcher`, so each canonical word inherits a measured time while the
   document stays authoritative for spelling.
3. **Interpolate** only the remainder, marking each such word `interpolated`.
4. **Cut cues** at measured pauses.

Three details that cost real time:

- **`autojunk` must be off.** It treats tokens in more than 1% of a long sequence as junk,
  which in a 2500-word section discards `the`, `to`, `of` and `a` — several hundred of the
  most reliable anchors in the text.
- **Anchors need support.** A size-1 matching block on `and` pairs unrelated occurrences; two
  such matches pinned 62 canonical words into 1.1 s. A run of three words is required, and any
  anchor pair implying more than 5 words/second is rejected (narration runs 2.5-3), dropping
  the weaker member so the region becomes an honest gap.
- **No second acoustic pass.** wav2vec2 forced alignment inside the recognised windows placed
  1764 of 2524 words against recognition's 2488, drift p90 4.03 s, and subdividing the
  failures made it worse. It discarded timings that were already correct.

### Text preparation carries most of the quality

`md-to-narration.py` strips what is not speech and rewrites what the voice reads wrong. Every
rule exists because its absence produced audible damage:

| rule | prevents |
|---|---|
| strip front matter, code, HTML comments, shortcodes | narrating art direction and metadata |
| handle `***bold italic***` before `**bold**`, then sweep any stray `*` | the voice reading "asterisk, asterisk, asterisk", then degenerating into `"asterisksisks"` |
| anchor `_italics_` to word boundaries; read `snake_case` as words; speak arrows as ", then" | `agency_account -> client -> source_export` fusing into `agencyaccount` and `sourceexport` |
| render tables row by row as short sentences | a run-on with no sentence boundaries; one produced 21 s of babble |
| unwrap `[ ]` task markers, `[placeholder]` and `<name>` | bracket punctuation read aloud, 34 times in one section |
| capitalise the first word of each paragraph | lowercase openers mispronounced — "complain" as "Dock and plane", 6 of 8 sampled wrong |
| space compounds the voice cannot say (`timezone`, `signup`) | "Heideheb" and "SignGen" |
| downcase SQL keywords and `SCREAMING_SNAKE` labels inside code spans | upper case switching the voice to spelling mode, inaccurately: `WHERE id = ?` as "WHAE IED", `OBSERVED_AT_IP, USED_DEVICE` as "OBS or VAT. ATIP, USE device" |
| split `CamelCase` identifiers to sentence case, not title case | "Order Cancellation Accepted" as "order, scancelation accepted"; the same phrase in sentence case is verbatim |
| verbalise operators, bound parameters, blanks and slashes in code spans | a sentence asserting nothing once `>` is dropped from `remaining > 0`; `___` as a run of underscores, which is the input most likely to start a repetition loop |
| place the currency unit by what follows the amount | `$4.8 million` as "4 dollars.8 million" — wrong, and a full stop the segmenter believes |
| read a semicolon as a sentence break; space compound hyphens | one table row reaching the voice as a single 218-character segment of eight noun phrases, rendered "QH, WEF codes, and paid boot and fulfilled orders" for "queue age, webhook retries, and paid-but-unfulfilled orders" |
| warn on surviving markup | all of the above shipping silently |

The last row is the important one. Upstream's FST-based text normalisation is not ported, so
these rules are doing that job by hand; a converter that reports what it could not handle is
the difference between finding these in minutes and finding them in a published file.

One known limitation, left unfixed deliberately. A clock time of the form `10:01` is spoken
correctly — the voice says "ten oh one" — but `10:00:02` comes out closer to "ten o'clock,
two", losing the seconds. It also makes the verifier noisy either way, because recognition
writes `10:01` as one token (`1001`) where the canonical text splits it into `10` and `01`, so
the span scores a low word overlap while the audio is right. In one book that was 41 of the
former against 13 of the latter, all inside incident timelines where the prose carries the
meaning, and the chapters were already rendered. If you are starting a book that leans on
timestamps, rewrite `h:mm:ss` in the converter before rendering — "10:01 and 4 seconds" reads
naturally and costs nothing up front. Do not retrofit it into a rendered book for the seconds
alone, and be careful changing the *tokenisation* to quiet the verifier: the page tokeniser
splits on a colon, and diverging from it is what once took a chapter from 100% page-mapped
to 18.5%.

The bottom half of that table was found by a method worth naming, because reading the
converted text will not find any of it: put the suspect constructs in one short passage,
render it, transcribe it back, and diff. Two minutes of audio settles questions that are
otherwise guesses — it distinguished a real defect from a sampling glitch twice (`B-tree` and
"granularity" both came back correct on a second draw), and it proved each fix by re-rendering
the identical clause. Do this before a long render, not after: the same evidence costs two
minutes up front or a full re-render later.

### Cue length is the number that matters

If the player interpolates the highlight across the active cue rather than sweeping word
timings, cue length bounds the visible error however precise those timings are. A cue must
also not span a pause, or words after the pause highlight during silence.

Because word times are measured, the pauses are visible and can be cut on: target ~3 s, break
at sentence punctuation or a gap over 220 ms, cap the maximum. Median lands at 2.4-3.0 s with
a worst case under 6 s.

Clamp forward afterwards. A player selecting a cue with `findIndex(t >= s.start && t < s.end)`
returns the *earlier* cue on an overlap and the highlight jumps backwards.

### Gates

Each manifest carries a `quality` block naming what failed rather than a number that always
looks healthy:

| gate | catches | observed healthy |
|---|---|---|
| measured share | words with no time from the audio | 96.8-99.5% |
| longest interpolated run | a passage recognition never found | 2-9 words |
| cue/pause agreement | a manifest shifted against the audio | 82-96% |
| page-word mapping | highlight cannot find words in the DOM | median 100%, worst 97.7% |

Cue/pause agreement is the one to add first, because it cannot share a failure mode with the
thing it checks:

```sh
ffmpeg -v info -i section.webm -af silencedetect=noise=-40dB:d=0.20 -f null -
```

Silence detection knows nothing about the script, the windows or the aligner. If cues were cut
where the voice pauses, their boundaries should land inside intervals ffmpeg independently
calls silence. A manifest can be perfectly self-consistent and still be shifted against the
audio; that shows up here and nowhere else.

Then verify the delivered files:

```sh
$ALIGN_PYTHON scripts/verify-narration.py narration/*.webm
```

It transcribes the closing seconds of each file — proving synthesis reached the end and the
deliverable decodes — then transcribes every span the aligner could not measure and compares
what was heard against what should have been said. Low overlap separates an audio defect from
a recognition miss: one needs a re-render, the other needs nothing.

### The free audit

**A word the voice mangles is never recognised, so it is already flagged `interpolated` in
every occurrence.** The manifests are therefore a defect report you have already paid for.

Aggregate them and ask which word types are unrecognised in at least 80% of appearances with
at least three occurrences. On a 146,000-word corpus that left about 37 suspects at no compute
cost.

Then listen to them, because most are innocent. `countermetric`, `tradeoff`, `codebase` and
`quickstart` are spoken correctly and merely transcribed as two words; `thirty` is transcribed
as `30`; `vale` and "Vail" are homophones. Six of the loudest signals were orthography
differences between recogniser and document rather than defects. Acting on the list without
listening triggers hours of re-rendering that fixes nothing.

### Delivery format

Delivery is **WebM/Opus at 48 kbps mono**, with the WAV kept beside it as the lossless master
so re-encoding never needs a re-render.

Bitrate, measured against the lossless WAV on a 17-minute section with faster-whisper WER as
the quality proxy:

| | size | WER |
|---|---|---|
| WAV (source) | 48 MB | 0.018 |
| MP3 128 kbps | 16 MB | 0.017 |
| Opus 32 kbps | 4.3 MB | 0.018 |
| Opus 24 kbps | 3.3 MB | 0.019 |

Opus is transparent by WER at 32 kbps, and 24 starts to cost something measurable. 48 kbps is
the default here for consistency with the sites this feeds and to leave headroom above the
transparency floor; `--bitrate 32k` is a defensible choice that halves the size.

**The container is a compatibility decision, not a detail.** Safari's support for Opus in an
Ogg container is unreliable, while Opus in WebM plays. Shipping `.opus` risks silent playback
failure on Safari that no amount of correct alignment would fix. Opus resamples to 48 kHz
internally; that is how the codec works and is not a quality loss.

### What a book costs

Measured on an M4 with 16 GB:

| | |
|---|---|
| source | 864,146 characters, 61 sections |
| audio | 16.1 hours |
| synthesis | ~11.9 hours at RTF 0.74 |
| WAV masters | 2.8 GB |
| delivery | ~336 MB |
| recognition | ~65 minutes total |

**Sample before committing to a full run.** Render two or three sections, run
`verify-narration.py` against them, fix what it surfaces, and only then start the rest. Nearly
every defect described above was visible in the first two sections; finding them one at a time
across five re-render rounds instead cost roughly eight hours of avoidable compute.


## Publishing into a static site

```sh
scripts/publish-narration.py --narration narration --site ../site --slug <book-slug>
```

`--dry-run` first; it prints exactly what it would place and change.

It installs each section as a site expecting this layout wants —
`static/audio/books/<slug>/<dir>/chapter.webm` with `manifest.json` beside it — and writes
`audio` and `audio_duration` into the section's front matter, which is typically what makes a
player render at all.

The manifest emits what a word-highlighting player needs: `delivery.segments[]` (`file`,
`durationSeconds`, `startSeconds`, `endSeconds`), `transcript.segments[]` (`start`, `end`,
`wordStart`, `wordEnd`) and `transcript.words[]` (`start`, `end`, `text`, optional
`pageWordIndex` and `pageWordNormalized`), plus the delivered container, codec and bitrate
probed from the file rather than assumed.

Three details that fail silently if missed:

- **`delivery.segments[].file` must be the published filename.** A player resolving it with
  `new URL(segment.file, manifestURL)` will render the page, load the manifest, and 404 the
  audio if the working name survives. The publisher rewrites it.
- **The manifest must sit beside the audio** when the template derives its URL from the audio
  path.
- **`pageWordNormalized` is worth shipping.** A player that checks it against the word it finds
  in the DOM can drop a mismatched index; omitting it to save bytes disables that defence, and
  a wrong index highlights the wrong word with nothing to notice it.

`chapter-N` becomes `chapter-NNN`; `introduction` and `conclusion` keep their names.

### Reusing it on another document

Nothing above is specific to one book. `--book DIR` handles both layouts; for anything else
pass files in reading order. `--engine audio8` switches voice, `--bitrate` the Opus rate,
`--seed` the sampling draw, `--only` a comma-separated list of sections, and `ALIGN_PYTHON`
the interpreter holding `faster-whisper` (autodetected when possible; its absence skips
manifests rather than failing the run).
