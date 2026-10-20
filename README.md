# Daily Job Search Agent

An automated, self-hosted job-search pipeline. Every day it scans Malaysian job
boards (JobStreet, LinkedIn, Workday, Greenhouse, Lever) for roles matching your
resume and target locations, scores the postings against your resume with
Gemini, and emails you a digest of the best matches — each with a **direct link
to the job posting**.

Built with **LangGraph**, **Google Gemini**, **Tavily Search**, and **SMTP**,
running on a free **GitHub Actions** cron.

## How it works

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌────────┐
│ planner │───▶│ searcher │───▶│ matcher │───▶│ mailer │
│ (Gemini)│    │ (Tavily) │    │ (Gemini)│    │ (SMTP) │
└─────────┘    └──────────┘    └─────────┘    └────────┘
  5 targeted     5 board-        score 0-100    HTML digest
  search         specific        vs resume,     with direct
  queries        queries,        keep ≥ 70%     posting URLs
                 URL + area
                 filtering
```

1. **Planner** — Gemini turns your roles/locations/resume into 5 targeted
   search queries (one per job board, location baked into each query). Falls
   back to deterministic templates if the LLM is down.
2. **Searcher** — runs each query through Tavily, then filters results with
   three layers:
   - *posting-shape check*: rejects category/salary/landing pages; only real
     posting URLs survive (e.g. `job-boards.greenhouse.io/<co>/jobs/<id>`,
     `my.jobstreet.com/job/<slug>-<id>`, LinkedIn `/jobs/view/…`)
   - *location check*: the posting must mention Kuala Lumpur (incl. Klang
     Valley / Petaling Jaya / Selangor / Cyberjaya …) or Penang (incl. Bayan
     Lepas / Kulim / Georgetwon …). Remote roles are only kept if `Remote`
     is explicitly in your target locations
   - *dedupe*: near-identical titles collapse to one entry
3. **Matcher** — Gemini scores each surviving posting 0–100 against your
   resume (in chunks of 5; unscored leftovers retried individually). Anything
   below `MATCH_THRESHOLD` (default 70) is dropped. Jobs the LLM couldn't
   score are filled in by a keyword-overlap heuristic so the digest still goes
   out (clearly labelled as degraded scoring).
4. **Mailer** — renders a self-contained HTML email with score badges, matched
   skills, gap skills, and a clickable **"View posting →"** button per job,
   then sends it over SMTP_SSL (Gmail app password).

## Setup (fork & go)

### 1. Secrets (required)

In your fork: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Value |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `TAVILY_API_KEY` | Tavily API key (free tier works) |
| `EMAIL_SENDER` | Gmail address that sends the digest |
| `EMAIL_APP_PASSWORD` | Gmail **app password** (16 chars; not your login password) |
| `EMAIL_RECIPIENT` | Where the digest is delivered |
| `RESUME_TEXT` | Your full resume as plain text (paste it in) |

Your resume never touches git — it lives only in the Actions secret. Optional
`RESUME_B64` (base64 of the resume) takes precedence over `RESUME_TEXT` if you
prefer a single-line value:

```powershell
# Windows PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("resume.txt"))
```

### 2. Variables (optional)

**Settings → Secrets and variables → Actions → Variables**

| Variable | Default | Meaning |
|---|---|---|
| `TARGET_ROLES` | `Backend Engineer, Machine Learning Engineer, Full Stack Developer` | Comma-separated roles to hunt |
| `TARGET_LOCATIONS` | `Kuala Lumpur, Malaysia; Penang, Malaysia` | `;`-separated locations; add `Remote` only if remote roles should be kept |

### 3. Run it

- **Scheduled**: every day at **00:00 UTC (08:00 MYT)** via cron.
- **Manual**: **Actions → Daily Job Search Agent → Run workflow**. You can
  override `roles` and `locations` per run without touching the config.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
python main.py
```

Useful flags:

```bash
python main.py --roles "Backend Engineer" --locations "Kuala Lumpur, Malaysia"
python main.py --resume path/to/resume.txt   # explicit file bypasses env secrets
```

Resume resolution order: `RESUME_B64` → `RESUME_TEXT` → local `resume.txt`
placeholder. Config resolution order for the run: CLI flags → env → defaults.

## Configuration reference

All variables (env file locally, secrets/vars on GitHub):

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `GEMINI_MODEL` | `gemini-3.8-flash` | auto-rotates to `gemini-3.5-flash` / `gemini-3.6-flash` on failure |
| `GEMINI_REQUEST_BUDGET` | `16` | hard cap on LLM calls per run |
| `TAVILY_API_KEY` | — | required |
| `EMAIL_SENDER` / `EMAIL_APP_PASSWORD` / `EMAIL_RECIPIENT` | — | required |
| `SMTP_HOST` / `SMTP_PORT` | `smtp.gmail.com` / `465` | SMTP_SSL |
| `TARGET_ROLES` | see above | |
| `TARGET_LOCATIONS` | see above | each entry expands to regional signals |
| `MATCH_THRESHOLD` | `70` | minimum score (0–100) to appear in the digest |
| `MAX_RESULTS_PER_QUERY` | `5` | Tavily depth per query |

## Resilience (what happens when things break)

Free-tier Gemini is flaky; the agent is built to still deliver a digest:

- **Model rotation** — a 404 (retired model) or daily-quota 429 moves the run
  to the next Gemini model; dead models are skipped for the rest of the run.
- **Transient errors** — 503 "high demand", 504 gateway timeouts, 500s and
  client read-timeouts are retried with a 15s backoff (up to 3 attempts per
  model).
- **Request budget** — at most 16 LLM calls per run, so one bad day can't
  burn the whole free-tier quota.
- **Chunked scoring** — postings are scored in chunks of 5; a failed chunk
  only loses those jobs, which then get individual retries.
- **Heuristic fallback** — if Gemini is fully unavailable, remaining postings
  are scored by keyword overlap between resume and posting. The email then
  carries a yellow **"degraded scoring"** banner and the subject says
  `(degraded scoring)` so you know the scores are approximate.
- **Deterministic queries** — if even the planner LLM is down, search still
  runs with template queries per role × location × board.

Note: Google's free-tier daily quota resets around midnight **Pacific** (≈07–08
UTC), so a run at 00:00 UTC can still see the previous day's quota. Steady-state
usage is only ~3–5 requests/day, well under the 20/day/model limit.

## Project layout

```
├── .github/workflows/daily_job_agent.yml   # cron + manual dispatch
├── main.py                                 # CLI entry, progress output, exit codes
├── src/
│   ├── config.py         # Settings + env loading
│   ├── state.py          # LangGraph state schema
│   ├── nodes.py          # planner / searcher / matcher / mailer nodes
│   ├── email_service.py  # HTML templates + SMTP send
│   └── graph.py          # LangGraph wiring
├── resume.txt            # placeholder (real resume comes from the secret)
├── requirements.txt
└── .env.example
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Email says "degraded scoring" | Gemini was down/quota'd; heuristic scores are stricter — treat <70 as "unknown", not "bad fit" |
| 0 matches but many postings found | Normal on placeholder resume; real `RESUME_TEXT` raises overlap |
| Run fails at config step | A required secret is missing (see Setup) |
| Gmail auth error | App password revoked — regenerate under Google Account → 2-Step Verification → App passwords |
| No Malaysian postings | Check `TARGET_LOCATIONS` and that queries bake the location in (planner log shows the 5 queries) |

## License

MIT — see [LICENSE](LICENSE).
