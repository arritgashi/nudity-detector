# Integration Guide

The moderation pipeline runs as a small HTTP service. Your application
posts an image to `POST /check` and gets back a verdict.

## Architecture

```
┌──────────────┐   multipart upload   ┌────────────────────────┐
│  Your app    │  ─────────────────▶  │  Moderation API        │
│  (any lang)  │                      │  (FastAPI on port 8000)│
│              │  ◀─────────────────  │  ┌──────────────────┐  │
└──────────────┘  JSON verdict        │  │ NudeNet (640m)   │  │
                                       │  │ Falconsai (NSFW) │  │
                                       │  └──────────────────┘  │
                                       └────────────────────────┘
```

Both services run on your own infrastructure. No external API calls,
no per-image cost, no images leave your network.

## Step 1 — Run the moderation service

Once-off (Windows):
```
install.bat        # if you haven't already
run-server.bat     # listens on http://0.0.0.0:8000
```

macOS/Linux:
```bash
./install.sh
./run-server.sh
```

Verify:
```bash
curl http://127.0.0.1:8000/health
# -> {"status":"ok","ready":true,"inference_resolution":1024}
```

## Step 2 — Test it with curl

```bash
./integrations/test-curl.sh path/to/test_image.jpg
```

You'll get back JSON like:

```json
{
  "verdict": "BLOCK",
  "reason": "Whole-image NSFW score 0.93 >= 0.7",
  "nsfw_score": 0.9341,
  "safe_score": 0.0659,
  "explicit_detections": [
    {
      "class": "MALE_GENITALIA_EXPOSED",
      "category": "explicit",
      "score": 0.6051,
      "box": [302, 145, 275, 247]
    }
  ],
  "all_detections": [...],
  "conflicts": [
    "Boosted 1 male-class score(s) by 2.50x to compensate for NudeNet bias"
  ],
  "elapsed_ms": 482,
  "model_versions": {
    "detector": "nudenet-640m",
    "classifier": "Falconsai/nsfw_image_detection"
  }
}
```

## Step 3 — Wire it into your application

Pick the file matching your stack:

| Stack | Example file |
|---|---|
| Node.js / Express | [node-express.js](node-express.js) |
| Next.js (App Router) | [nextjs-app-router.ts](nextjs-app-router.ts) |
| Python / Django | [python-django.py](python-django.py) |
| PHP / Laravel | [php-laravel.php](php-laravel.php) |

Each example shows the full pattern: receive upload → call /check → branch on verdict (ALLOW/REVIEW/BLOCK) → store or reject.

## Endpoints

### `GET /health`
Returns `{"status":"ok","ready":true,"inference_resolution":1024}` when the models are loaded. Use this in your application's deployment health check.

### `GET /policy`
Returns the active thresholds. Useful for debugging "why was X blocked?".

### `POST /check`
Multipart upload, field name `file`. Returns the JSON shown above.

Only JPEG is accepted (`MOD_FORMATS`), one frame, at most `MOD_MAX_SIDE` pixels a side (2048): other formats get 415, larger images 413.
`POST /check_url` was removed: it could be pointed at addresses inside the server.

## Verdict semantics

The service returns one of three verdicts:

- **`ALLOW`** — image is safe. Your app stores it as the profile picture.
- **`REVIEW`** — borderline. Your app stores it but marks for human moderation. (You decide whether to show it immediately or hold until reviewed.)
- **`BLOCK`** — image is explicit. Your app rejects with an error message.

## Configuration via environment variables

All thresholds are tunable without editing code:

| Variable | Default | Effect |
|---|---|---|
| `MOD_NSFW_BLOCK` | `0.70` | Falconsai score >= this → BLOCK |
| `MOD_NSFW_REVIEW` | `0.30` | Falconsai score >= this → REVIEW |
| `MOD_EXPLICIT_BLOCK` | `0.50` | Any explicit body part >= this → BLOCK |
| `MOD_EXPLICIT_REVIEW` | `0.20` | Any explicit body part >= this → REVIEW |
| `MOD_MALE_BOOST` | `2.5` | Male-class score multiplier |
| `MOD_INFERENCE_RESOLUTION` | `1024` | Detector input size |
| `MOD_DISPLAY_THRESHOLD` | `0.15` | Min score to include detection in response |
| `MOD_MAX_IMAGE_BYTES` | `15728640` | Reject uploads larger than this |
| `MOD_ALLOWED_ORIGINS` | `*` | CORS — comma-separated list of allowed origins |

Set them in the environment before launching the service:

```bash
MOD_NSFW_BLOCK=0.6 MOD_MALE_BOOST=3.0 ./run-server.sh
```

## Production deployment

### Same-machine (simplest)
Run the moderation service alongside your app on the same box. Your app calls `http://127.0.0.1:8000/check`. Use a process supervisor (systemd unit, pm2, Windows Service, supervisord) to keep it up.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud
- **Render / Fly.io / Railway** — deploy the folder as a service, ~$7/mo
- **Hetzner / DigitalOcean** — $5/mo VPS handles thousands of checks per day
- **Hugging Face Spaces** — free, but only for the Gradio UI; not suitable for the API

### Scaling
- Single CPU process handles ~2-5 images/sec on a modern server
- For higher throughput: run multiple uvicorn workers (`--workers 4`)
- For very high throughput: GPU. Falconsai is ~10x faster on GPU.

## Failure modes to plan for

| Failure | Symptom | What to do |
|---|---|---|
| Service down | Your app gets timeout/connection refused | **Fail-closed** (reject upload) — included in all examples. Don't fail-open. |
| Service slow | First request after restart takes ~10s | Warm up at deploy time: `curl /health` after startup |
| Model wrong | False positive on innocent image | Add an "appeal" flow — let user request human review. Adjust `MOD_NSFW_BLOCK` upward. |
| Model wrong | False negative on bad image | Lower `MOD_NSFW_BLOCK` and `MOD_MALE_BOOST` — catches more but rejects more. |

## Auto-generated API docs

Once the server is running, visit **http://127.0.0.1:8000/docs** for an interactive Swagger UI you can test from the browser.
