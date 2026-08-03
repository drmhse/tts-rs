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
