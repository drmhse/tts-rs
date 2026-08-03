//! `tts-serve` — the HTTP surface, wire-compatible with the Python service it replaces.
//!
//! The Python service (`CosyVoice/serve.py`, uvicorn on `PORT`, default 3003) is the
//! contract clients already speak, so this reproduces it rather than inventing a new one:
//! same paths, same request bodies, same response headers, same auth. Point a client at
//! this instead and it should not notice, except in latency.
//!
//! # What this does not carry over, and why
//!
//! Most of that service's complexity exists to work around PyTorch. `run.sh` sets
//! `TTS_WORKER_MAX_GROWTH_MB`, `MAX_FOOTPRINT_MB`, `MAX_REQUESTS` and `IDLE_SECONDS`, and
//! its own comment says why: *"PyTorch's MPS backend never frees its compiled-graph cache,
//! so ending the process is the only way to reclaim it."* Hence a subprocess worker, a
//! recycle budget, a supervisor, and a ~15 s reload whenever the budget trips.
//!
//! None of that applies here. There is no MPSGraph cache, memory is flat by construction,
//! and both engines are held resident in this process for its lifetime. So the supervisor,
//! the worker, `modelWorker` bookkeeping and the reload cost are all simply gone — the
//! endpoints that reported on them still exist and answer honestly, they just have much
//! less to say.
//!
//! Two things are **not implemented** and say so with `501` rather than pretending:
//! the durable job queue (`/v1/tts-jobs`) and forced alignment (`/v1/alignment-jobs`).
//! Alignment in particular runs a separate whisper environment in the Python service; it
//! is a subprocess call from here, not a port. `GET /` lists what is live.
//!
//! # Concurrency
//!
//! One GPU, so synthesis is serialised behind a permit rather than run concurrently —
//! two requests interleaving on one Metal queue would make both slower and neither
//! faster. Requests queue; the semaphore is the queue. Synthesis itself is blocking and
//! runs on `spawn_blocking` so it never occupies an async worker.

use anyhow::{Context, Result};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;
use tts_core::{Engine, EngineConfig, Sampling, SynthesisRequest, Voice};

/// The Python service's default. Overridable the same way: `PORT=…`.
const DEFAULT_PORT: u16 = 3003;

#[derive(Parser)]
#[command(
    name = "tts-serve",
    about = "HTTP TTS, wire-compatible with the Python service"
)]
struct Args {
    /// Defaults to `$PORT`, then 3003.
    #[arg(long)]
    port: Option<u16>,
    /// Loopback by default. A model this size should not be exposed casually.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Engine id; `tts engines` lists them.
    #[arg(long, default_value = "cosyvoice")]
    engine: String,
    /// Voice asset directory used when a request does not name one.
    #[arg(long, default_value = "voices/cosy-default-cosyvoice")]
    voice: String,
    #[arg(long)]
    model_root: Option<String>,
    #[arg(long)]
    quant: Option<String>,
    #[arg(long)]
    cpu: bool,
    /// Per-request character ceiling, mirroring the Python service's `TTS_MAX_CHARS`.
    #[arg(long, default_value_t = 1200)]
    max_chars: usize,
    /// Segment budget handed to the engine's own segmenter.
    #[arg(long, default_value_t = 220)]
    segment_chars: usize,
    /// Fallback when `TTS_API_KEY` is unset, mirroring the Python service's `.api_key`.
    #[arg(long, default_value = ".api_key")]
    api_key_file: String,
}

struct App {
    engine: Box<dyn Engine>,
    voice: Voice,
    engine_id: String,
    sample_rate: u32,
    max_chars: usize,
    segment_chars: usize,
    api_key: Option<String>,
    /// One GPU: synthesis takes this before running.
    gpu: tokio::sync::Semaphore,
    started: Instant,
}

// ---------------------------------------------------------------- errors

/// An error that serialises the way FastAPI's `HTTPException` does, so a client's
/// error handling does not have to change either.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "detail": self.1 }))).into_response()
    }
}

fn bad(code: StatusCode, msg: impl Into<String>) -> ApiError {
    ApiError(code, msg.into())
}

// ---------------------------------------------------------------- auth

/// `Authorization: Bearer <key>` or `X-API-Key: <key>`, matching the Python service.
///
/// Compared with a length-checked constant-time equality rather than `==`, for the same
/// reason `serve.py` reaches for `secrets.compare_digest`.
fn require_key(app: &App, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = app.api_key.as_deref() else {
        // The Python service refuses rather than running open. Same here: a missing key
        // is a misconfiguration, and defaulting to "no auth" is the wrong guess.
        return Err(bad(
            StatusCode::SERVICE_UNAVAILABLE,
            "server API key not configured — set TTS_API_KEY",
        ));
    };
    let provided = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let (scheme, rest) = v.split_at(v.len().min(7));
            scheme.eq_ignore_ascii_case("bearer ").then(|| rest.trim())
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .map(str::trim)
        });

    match provided {
        Some(got) if constant_time_eq(got.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err(ApiError(
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key".into(),
        )),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ---------------------------------------------------------------- request body

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TtsRequest {
    text: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default)]
    instruct_text: Option<String>,
    #[serde(default = "default_speed")]
    speed: f32,
    /// Not in the Python schema. Additive, so an existing client is unaffected: it names
    /// a voice asset directory per request instead of using the one this was started with.
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
}

fn default_mode() -> String {
    "zero_shot".into()
}
fn default_speed() -> f32 {
    1.0
}

/// The validation `serve.py::_validate` does, plus the knobs this port cannot honour.
///
/// Rejecting rather than ignoring is deliberate and is the rule the engine trait already
/// states: an engine documents the controls it ignores instead of silently accepting them.
/// Quietly returning `speed: 1.0` audio to a client that asked for 1.5 is worse than a 501,
/// because nothing in the response says the request was not honoured.
fn validate(app: &App, req: &TtsRequest) -> Result<(), ApiError> {
    if req.text.trim().is_empty() {
        return Err(bad(StatusCode::BAD_REQUEST, "text is empty"));
    }
    if req.text.chars().count() > app.max_chars {
        return Err(bad(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "text too long: {} chars > max_chars={}. Split into multiple requests.",
                req.text.chars().count(),
                app.max_chars
            ),
        ));
    }
    match req.mode.as_str() {
        "zero_shot" => {}
        m @ ("instruct" | "cross_lingual") => {
            return Err(bad(
                StatusCode::NOT_IMPLEMENTED,
                format!(
                    "mode='{m}' is not implemented in the Rust port — only 'zero_shot'. \
                     The port has no instruction-prompt path; see docs/porting/cosyvoice.md."
                ),
            ))
        }
        other => {
            return Err(bad(
                StatusCode::BAD_REQUEST,
                format!("unknown mode '{other}'"),
            ))
        }
    }
    if (req.speed - 1.0).abs() > f32::EPSILON {
        return Err(bad(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "speed={} is not implemented — the port synthesizes at 1.0 only, and \
                 returning unmodified audio would misreport the request as honoured.",
                req.speed
            ),
        ));
    }
    if req.instruct_text.is_some() {
        return Err(bad(
            StatusCode::NOT_IMPLEMENTED,
            "instruct_text is not implemented (mode='instruct' is unsupported)",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- synthesis

struct Rendered {
    wav: Vec<u8>,
    seconds: f64,
    wall: f64,
    stages: Vec<(&'static str, f64)>,
    samples: Vec<f32>,
    sample_rate: u32,
}

async fn render(app: &Arc<App>, req: TtsRequest) -> Result<Rendered, ApiError> {
    let voice = match &req.voice {
        None => app.voice.clone(),
        Some(dir) => Voice::load(dir)
            .map_err(|e| bad(StatusCode::BAD_REQUEST, format!("loading voice {dir}: {e}")))?,
    };

    let text = req.text.clone();
    let seed = req.seed;

    // One GPU: queue rather than contend. Held across the blocking call below, so the
    // permit — not the thread pool — is what bounds concurrent synthesis.
    let _permit = app
        .gpu
        .acquire()
        .await
        .map_err(|_| bad(StatusCode::SERVICE_UNAVAILABLE, "server shutting down"))?;

    let app2 = Arc::clone(app);
    let started = Instant::now();
    let out = tokio::task::spawn_blocking(move || {
        let mut request = SynthesisRequest::new(text).with_voice(voice);
        request.max_chars = app2.segment_chars;
        if let Some(s) = seed {
            request.sampling = Sampling {
                seed: s,
                ..request.sampling
            };
        }
        app2.engine.validate(&request)?;
        app2.engine.synthesize(&request)
    })
    .await
    .map_err(|e| {
        bad(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("worker panic: {e}"),
        )
    })?
    .map_err(|e| bad(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;

    let wall = started.elapsed().as_secs_f64();
    let seconds = out.audio.seconds();
    // One encoder, in tts-core: a second copy here silently disagreed with the CLI
    // by 1 LSB because it truncated where that one rounds.
    let wav = tts_core::wav::to_bytes(&out.audio);
    Ok(Rendered {
        wav,
        seconds,
        wall,
        stages: out.stats.stages,
        samples: out.audio.samples,
        sample_rate: out.audio.sample_rate,
    })
}

// ---------------------------------------------------------------- handlers

async fn post_tts(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(req): Json<TtsRequest>,
) -> Result<Response, ApiError> {
    require_key(&app, &headers)?;
    validate(&app, &req)?;
    let r = render(&app, req).await?;

    let hdr = |n: &'static str, v: String| (HeaderName::from_static(n), HeaderValue::from_str(&v));
    let mut out = Response::builder()
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::CONTENT_DISPOSITION, "inline; filename=\"tts.wav\"");
    for (name, value) in [
        hdr("x-audio-seconds", format!("{:.2}", r.seconds)),
        hdr("x-wall-seconds", format!("{:.2}", r.wall)),
        hdr(
            "x-rtf",
            format!(
                "{:.2}",
                if r.seconds > 0.0 {
                    r.wall / r.seconds
                } else {
                    0.0
                }
            ),
        ),
        hdr("x-audio-format", "pcm_s16le_mono".into()),
        // Additive: the per-stage split the CLI prints, so a client can see where the
        // time went without a second request.
        hdr(
            "x-stages",
            r.stages
                .iter()
                .map(|(n, s)| format!("{n}={s:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        ),
    ] {
        if let Ok(v) = value {
            out = out.header(name, v);
        }
    }
    out.body(r.wav.into())
        .map_err(|e| bad(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Raw little-endian 16-bit mono PCM, as the Python service's `/tts/stream` emits.
///
/// **Honest limitation:** the Python service streams from the model as it decodes, so its
/// point is time-to-first-audio. Neither engine here exposes an incremental decode yet
/// (`Capabilities::streaming` is false for both), so this synthesizes fully and then
/// writes the body. The bytes on the wire are identical and a client needs no change —
/// but it does not get the latency benefit, and saying otherwise would be a lie the
/// header cannot carry.
async fn post_tts_stream(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(req): Json<TtsRequest>,
) -> Result<Response, ApiError> {
    require_key(&app, &headers)?;
    validate(&app, &req)?;
    let r = render(&app, req).await?;
    let body = tts_core::wav::pcm_s16le(&r.samples);
    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header("x-sample-rate", r.sample_rate.to_string())
        .header("x-audio-format", "pcm_s16le_mono")
        .header("x-streaming", "buffered")
        .body(body.into())
        .map_err(|e| bad(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

async fn get_root(State(app): State<Arc<App>>) -> Json<serde_json::Value> {
    Json(json!({
        "service": "tts-rs",
        "engine": app.engine_id,
        "sample_rate": app.sample_rate,
        "max_chars": app.max_chars,
        "uptime_seconds": app.started.elapsed().as_secs(),
        "endpoints": ["/health", "/v1/capabilities", "POST /tts", "POST /tts/stream"],
        "not_implemented": {
            "POST /v1/tts-jobs": "durable job queue",
            "POST /v1/alignment-jobs": "forced alignment (needs a whisper environment)",
            "GET /v1/artifacts/{job_id}/{filename}": "job artifacts",
        },
        "guide": "https://github.com/drmhse/tts-rs",
    }))
}

async fn get_health(State(app): State<Arc<App>>) -> Response {
    // Models are loaded before the listener binds, so if this answers at all it is ready.
    // The Python service can be up-but-loading because its worker starts lazily.
    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "model_loaded": true,
            "engine": app.engine_id,
            "device": if cfg!(feature = "metal") { "metal" } else { "cpu" },
            "uptime_seconds": app.started.elapsed().as_secs(),
        })),
    )
        .into_response()
}

async fn get_capabilities(State(app): State<Arc<App>>) -> Json<serde_json::Value> {
    let caps = app.engine.capabilities();
    Json(json!({
        "apiVersion": "v1",
        "service": "tts-rs",
        "engine": caps.id,
        "description": caps.description,
        "sampleRate": caps.sample_rate,
        "frameRate": caps.frame_rate,
        "quantization": caps.quantization,
        "maxCharsPerRequest": app.max_chars,
        "maxCharsPerSegment": app.segment_chars,
        "modes": ["zero_shot"],
        "audio": {"container": "wav", "encoding": "pcm_s16le", "channels": 1},
        "streaming": caps.streaming,
        "cloning": format!("{:?}", caps.cloning),
        "jobs": {"durable": false, "reason": "not implemented in the Rust service"},
        "alignment": {"available": false, "reason": "needs a separate whisper environment"},
        // No supervisor, no recycle budget, no reload: there is no MPSGraph cache to
        // reclaim, so the model stays resident for the process lifetime.
        "modelWorker": {"inProcess": true, "recycles": 0, "reason": "memory is flat by construction"},
    }))
}

async fn not_implemented(Path(rest): Path<String>) -> ApiError {
    bad(
        StatusCode::NOT_IMPLEMENTED,
        format!(
            "/v1/{rest} is not implemented by tts-serve. The durable job queue, forced \
             alignment and artifact retrieval live only in the Python service. GET / lists \
             what this one serves."
        ),
    )
}

// ---------------------------------------------------------------- main

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let port = args
        .port
        .or_else(|| std::env::var("PORT").ok()?.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    // Same resolution order as `serve.py::_load_api_key`: the environment, then a
    // `.api_key` file beside the service. Matching it means an existing deployment can
    // point at this binary without moving its secret.
    let api_key = std::env::var("TTS_API_KEY")
        .ok()
        .or_else(|| std::fs::read_to_string(&args.api_key_file).ok())
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());
    if api_key.is_none() {
        eprintln!(
            "warning: no API key found (checked $TTS_API_KEY and {}) — every authenticated\n\
             \x20        route will answer 503, exactly as the Python service does. Point\n\
             \x20        --api-key-file at the existing deployment's .api_key so clients\n\
             \x20        need no change.",
            args.api_key_file
        );
    }

    let root = args
        .model_root
        .clone()
        .unwrap_or_else(|| tts_engines::default_root(&args.engine).to_string());
    let config = EngineConfig {
        model_root: root.into(),
        quant: args.quant.clone(),
        cpu: args.cpu,
        overrides: BTreeMap::new(),
    };

    eprintln!(
        "loading engine `{}` from {:?}…",
        args.engine, config.model_root
    );
    let load = Instant::now();
    let engine = tts_engines::load(&args.engine, &config)
        .with_context(|| format!("loading engine {}", args.engine))?;
    let voice =
        Voice::load(&args.voice).with_context(|| format!("loading voice asset {}", args.voice))?;
    let caps = engine.capabilities();
    eprintln!(
        "loaded in {:.2}s — {} at {} Hz",
        load.elapsed().as_secs_f64(),
        caps.id,
        caps.sample_rate
    );

    let app = Arc::new(App {
        engine_id: caps.id.to_string(),
        sample_rate: caps.sample_rate,
        engine,
        voice,
        max_chars: args.max_chars,
        segment_chars: args.segment_chars,
        api_key,
        gpu: tokio::sync::Semaphore::new(1),
        started: Instant::now(),
    });

    let router = Router::new()
        .route("/", get(get_root))
        .route("/health", get(get_health))
        .route("/v1/capabilities", get(get_capabilities))
        .route("/tts", post(post_tts))
        .route("/tts/stream", post(post_tts_stream))
        // Answer the unimplemented v1 surface explicitly. A 404 would look like a wrong
        // URL; a 501 says the route is real and this server does not serve it.
        .route("/v1/*rest", get(not_implemented).post(not_implemented))
        .with_state(Arc::clone(&app));

    let addr = format!("{}:{}", args.host, port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr} — is the Python service still on this port?"))?;
    eprintln!("listening on http://{addr}");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("\nshutting down");
        })
        .await?;
    Ok(())
}
