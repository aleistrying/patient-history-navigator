# Public deployment on Vercel

**Live: <https://patient-history-navigator.vercel.app>** — password-protected; ask the team for the password.

Puts the navigator behind a shareable URL so a reviewer can use it without a
Databricks login. The browser never holds a Databricks credential: the page calls
`/api/ask`, and that serverless function adds the bearer token from an encrypted
environment variable before forwarding to Model Serving.

```
browser ──POST /api/ask──▶ Vercel function ──Bearer token──▶ Model Serving endpoint
   ▲                            (env vars, encrypted)                │
   └──────────────── answer + sources + tool_data ───────────────────┘
```

## The three functions

| Route | What it does |
| --- | --- |
| `api/ask.py` | Checks the password, signs the invocation, returns the answer, its sources and the structured tool output. |
| `api/patients.py` | Serves the patient roster. Touches nothing sensitive. |
| `api/warm.py` | Fires one short-timeout probe to start the endpoint scaling up, and reports `ready` or `warming`. Backs the **Wake it now** button. |

**These must stay in step with `ui/app.py`, which serves the same three routes
locally.** The wake bar once reported "still waking" forever on the deployed copy
because `api/warm.py` did not exist: the route 404'd, the client could not parse
the error body, and it read that as "not ready yet". A route missing from one of
two deployments is invisible to any check run against the other.

`/api/warm` takes the password as a query parameter rather than a body, because it
is a GET. `sync_from_ui.py` adds that parameter and asserts it is present.

## Deploy

```bash
npm i -g vercel               # once
cd vercel
python3 sync_from_ui.py       # regenerate index.html from ui/index.html
vercel                        # first run links the project
vercel --prod                 # publish
```

Run these **from inside `vercel/`**, not from the repository root. Vercel serves
functions from an `api/` directory at the deployment root, and this project's
functions are `vercel/api/`, so a project linked one level up deploys a page with
no working API. The `.vercel/` link directory therefore lives here.

Each `vercel --prod` prints a one-off deployment URL
(`patient-history-navigator-<hash>.vercel.app`). That is not the address to share:
it changes on every deploy. The stable production alias is
`patient-history-navigator.vercel.app`, and `--prod` repoints it automatically.

## Two things that will make it look broken

**Deployment Protection.** A new Vercel project can have SSO protection on by
default, which redirects every visitor to a Vercel login (HTTP 302 to
`vercel.com/sso-api`) — including the professor. Turn it off under
**Settings → Deployment Protection**; the app's own password gate is what
restricts access here.

**Environment variables need a redeploy.** Changing one in the dashboard does not
affect running functions. Run `vercel --prod` again afterwards, or the old value
stays live.

## Environment variables

Set these in **Vercel → Project → Settings → Environment Variables** (never in the
repository). Add each to Production, Preview and Development.

| Name | Value | Why |
| --- | --- | --- |
| `DATABRICKS_HOST` | `https://dbc-b7a36bd8-4a4a.cloud.databricks.com` | workspace to call |
| `DATABRICKS_TOKEN` | a personal access token | signs the request |
| `PHN_PASSWORD` | a long passphrase, shared out of band | stops strangers spending your compute |
| `PHN_ENDPOINT` | `dbdemos-rag-prod` | optional, this is the default |

Create the token with:

```bash
databricks tokens create --comment "vercel patient history navigator" --lifetime-seconds 7776000
```

The token needs **CAN QUERY** on the serving endpoint, and for `/api/warm` to rebuild
missing endpoints, the right to create serving and AI Search endpoints. A personal
access token carries its owner's permissions, which covers both. Treat it as a
secret: it can run queries against your workspace and consume Free Edition compute.

## Why a password, and why it is not in this repository

The URL is public, so without one anybody who finds it can run queries billed to
your workspace. The records themselves are synthetic, so the exposure is cost
rather than privacy, but the gate costs one line to add and removes the question.
The page stores the password in `sessionStorage`, so a reviewer types it once.

The repository is public and the page can create Databricks resources (see
`api/warm.py`), so the password lives only in the Vercel environment and is shared
directly with whoever needs it. The one in this repository's early history is no
longer valid. To change it, without it ever touching shell history or a file here:

```bash
cd vercel
vercel env rm PHN_PASSWORD production --yes
vercel env add PHN_PASSWORD production   # prompts for the value
vercel --prod        # required: functions keep the old value until redeployed
```

`verify_deployment.py` reads the password from the `PHN_PASSWORD` environment
variable, so running the check does not need it written anywhere here.

## Timing, and why `maxDuration` is 300

`vercel.json` sets `maxDuration: 300` on `/api/ask`. That is deliberate:

| Situation | Observed |
| --- | --- |
| Endpoint scaled to zero, first call | ~55 s |
| Retrieval answer, warm | 15–16 s |
| Tool answer, warm | 4–10 s |

Model Serving scales to zero after roughly 10 minutes idle, so the first question
after a quiet period pays the wake-up. Vercel's Hobby plan allows up to 300 s with
fluid compute, which covers it comfortably. Send one throwaway question before a
demonstration and every subsequent answer is fast.

## Keeping it warm (optional)

A Vercel cron hitting the endpoint every few minutes avoids the cold start, at the
cost of keeping Model Serving running. Add to `vercel.json`:

```json
"crons": [{ "path": "/api/patients", "schedule": "*/5 * * * *" }]
```

Note that `/api/patients` does not touch Model Serving, so it will **not** keep the
endpoint warm — point a cron at a real question only if you accept the compute cost.

## What is not here

The page is the same one served locally by `ui/app.py`, plus a password gate. Keep
them in step: the canonical copy is `ui/index.html`, and `vercel/index.html` is that
file with the gate applied.
