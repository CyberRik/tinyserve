# TinyServe Demo Page — PRD

**Status:** Design doc for a recording/demo surface, not a product feature
**Scope:** one static HTML page served by the existing FastAPI app at `GET /demo`
**Companion doc:** [`demo-recording-guide.md`](./demo-recording-guide.md)

---

## 1. Why this exists

`docs/demo-recording-guide.md` originally specified five README demos, four of them terminal
recordings. That shape had a good reason behind it: the claims TinyServe actually makes — tokens from independent
requests interleaving in one batch, a burst getting rejected at the door, one scheduling policy
treating priority differently from another — are *live, concurrent, multi-request* behaviours.
Two curl panes show that honestly. Swagger UI cannot: it buffers the response body, so
token-by-token streaming never renders, and it fires one request at a time from a form, so
concurrency and admission pressure are invisible.

The gap is that "watch two terminal panes" is a high-context ask for a README visitor, and the
project's most interesting property — *many* requests sharing one batch loop — gets harder to
show the more requests you add. You cannot open 32 terminal panes.

This page closes that gap. It is a client, not a feature: it drives the same public
`POST /generate` endpoint the curl demos drive, over the same SSE stream, and renders N
concurrent streams side by side with per-request timing. It shows more than a terminal can,
without staging anything a terminal could not also prove.

**The honesty constraint that shapes every decision below:** the page must not compute, smooth,
or invent any number it did not observe on the wire. Every figure it displays is either a
timestamp it took itself around a real `fetch`, or a value scraped verbatim from `/metrics`.

## 2. Goals

- Show **continuous batching** at a request count a terminal cannot display — 8–32 concurrent
  streams, tokens visibly arriving interleaved rather than one request completing at a time.
- Show **admission control** as a visible event: a burst large enough to trip the KV-budget or
  queue-depth check renders rejected requests as rejected cards, tagged with the server's own
  `reason` string, not a generic error.
- Show **priority** mattering: a mixed-priority burst under `wfq` or `priority` should make the
  high-priority cards visibly reach first token earlier than low-priority ones.
- Be **recordable**: a fixed layout, a dark palette matching the recording guide's Catppuccin
  Mocha terminal theme, and no motion that depends on mouse position — so a screen capture of
  this page sits beside the terminal GIF as one visual family.
- Compare **scheduling policies on one server**, by hot-swapping the policy between otherwise
  identical bursts — replacing the previous three-servers-on-three-ports setup that
  `benchmarks/fairness.py` documents.
- Show **what actually happened** in a timestamped event log, so a viewer reading a still
  frame can see request lifecycles and full response text, not just animated cards.
- Be **reproducible by a visitor**: served from the runtime itself, so `uv run uvicorn ...` then
  `localhost:8000/demo` is the entire setup.

## 3. Non-goals

- **Not a production console.** No auth, no persistence, no historical view, no per-request
  drill-down. Grafana already owns real observability; this owns *one burst, watched live*.
- **Not a replacement for the terminal demo.** A single stream rendering token by token is
  still better as a 10s `curl -N` GIF — it needs zero context and no UI to explain. That
  survives as Demo 3 in the guide. This page replaces the old Demos 1 and 3 (concurrency,
  and the FIFO-vs-WFQ comparison), which between them needed tmux and two extra servers.
- **Not a benchmark.** Numbers on this page are browser-side observations of a handful of
  requests, subject to browser connection limits (§7). `docs/benchmarks.md` remains the only
  source of published performance numbers, and the page says so on its face.
- **No build step.** No npm, no bundler, no framework. One self-contained `.html` file with
  inline CSS and JS. A build toolchain for one demo page would be the most disproportionate
  dependency in the repo.
- **No new *scheduling* behaviour.** The page adds three routes (`GET /demo`, `GET /config`,
  `POST /config/policy`) and one level of indirection in front of the active policy. It adds
  no policy, changes no admission or KV logic, and alters no batch-loop arithmetic. See §5.4
  for the one real behavioural change and its cost.

## 4. Why it is served from FastAPI, not opened as a file

The app has no CORS middleware. A page opened from `file://` or a separate static server is a
different origin, so every `fetch` to `127.0.0.1:8000/generate` fails the preflight and the page
shows nothing.

Two ways out: add `CORSMiddleware`, or serve the page from the app itself. This PRD chooses
serving it, because adding permissive CORS to the runtime would loosen the security posture of
the actual server for the benefit of a demo — a bad trade for a project whose README argues for
deliberate scope lines. Serving one static file at `/demo` adds no cross-origin surface at all,
and makes the demo's own URL (`localhost:8000/demo`) proof that it is talking to the real
runtime rather than a mock.

**Cost, stated plainly:** this is a small runtime change (one route + one static file in the
package), not zero.

## 5. The surface

```
┌────────────────────────────────────────────────────────────────────────────┐
│ TinyServe   policy [wfq ▾]      n_seq_max 2 · queue 3 · kv 41/128 · 12 / 4  │ ← /config + /metrics
├────────────────────────────────────────────────────────────────────────────┤
│ requests[16] max_tokens[24] mix[3:1 ▾] prompt[…]                            │
│                         [ Send burst ] [ Compare policies ] [ Reset ]       │ ← controls
├────────────────────────────────────────────────────────────────────────────┤
│ accepted 12 · rejected 4 · streaming 9 · done 3 · 247 tok · 4.2s            │ ← tallies
├────────────────────────────────────────────────────────────────────────────┤
│ policy    accepted  rejected  low p1     high p5    ratio                   │
│ fifo      16        0         3108ms     3775ms     0.82x                   │ ← compare table
│ priority  16        0         4395ms     1113ms     3.95x                   │   (after Compare)
│ wfq       16        0         4020ms     1523ms     2.64x                   │
├────────────────────────────────────────────────────────────────────────────┤
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐                    │
│ │#3 p5  ▓▓▓ │ │#4 p1  ▓▓  │ │#5 p1  503 │ │#6 p5 ▓▓▓▓ │                    │ ← stream grid
│ │ttft 210ms │ │ttft 480ms │ │kv_cache_  │ │ttft 190ms │                    │
│ │The story… │ │A detectiv…│ │full       │ │Two friend…│                    │
│ └───────────┘ └───────────┘ └───────────┘ └───────────┘                    │
├────────────────────────────────────────────────────────────────────────────┤
│ EVENT LOG                                              [ Copy ] [ Clear ]   │
│ [  0.004s] burst: 16 requests, max_tokens=24, policy=wfq, mix=3:1           │ ← event log
│ [  0.213s] #3 first token @ 210ms (p5)                                      │
│ [  0.319s] #5 rejected 503 kv_cache_full                                    │
│ [  2.104s] #3 done — 24 tok in 1.89s (12.7 tok/s)                           │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Controls

| Control | Range | Purpose |
|---|---|---|
| requests | 1–64 | burst size; the lever that makes batching vs. rejection visible |
| max_tokens | 8–128 | per-request generation length |
| priority mix | `all low` / `3:1` | `3:1` low:high using priorities **1 and 5**, matching `benchmarks/fairness.py` exactly |
| policy | `fifo`/`priority`/`wfq` | hot-swaps the live policy (§5.4) |
| Send burst | — | fires all N concurrently, then streams |
| Compare policies | — | same burst under all three policies, tabulated (§5.6) |
| Reset | — | clears the grid and table without reloading |

### 5.2 Per-request card

Each card shows: short id, priority badge (p1 / p5), state, TTFT, token count, tokens/sec, and
the live text. States are `queued` (request sent, no first token yet), `streaming`, `done`, `rejected`
(503, showing the server's `reason`), and `error`.

### 5.3 Live server strip

Reads `GET /config` once at load for policy / `n_seq_max`, then polls `GET /metrics` every
second for `queue_depth`, `kv_blocks_used`/`kv_blocks_free`, and the accept/reject counters — scraped verbatim from the same endpoint Prometheus scrapes.
This is what ties the page to the Grafana demo: same numbers, same source, different lens.

### 5.4 Policy hot-swap — the one real runtime change

`POST /config/policy` swaps the scheduling policy on a running server. The batch loop reads the
policy through an `ActivePolicy` holder each tick instead of closing over one object at startup.

**Why it earns its place:** `benchmarks/fairness.py` needs three servers on three ports to
compare policies, because the policy was fixed at startup. That is a fine shape for a benchmark
script and a bad one for a demo — nobody watching a GIF wants to see three terminals. With the
swap, the comparison is one button on one server.

**What it costs, stated plainly:**
- It is an **unauthenticated mutation endpoint**. Every route here is unauthenticated and the
  server binds to localhost, so this changes the posture less than it might appear — but it is
  the first route that changes server behaviour rather than reporting it. It must not be exposed
  publicly, and the docstring says so.
- **Swaps are not retroactive.** Sequences already holding a concurrency slot run to completion
  under the policy that admitted them; only later admissions see the new policy. The comparison
  flow therefore waits for `queue_depth` to reach 0 between runs rather than swapping mid-burst.
- It adds a mutable field to a loop that was previously pure in this respect. Mitigated by
  keeping the mutation to one place (`ActivePolicy.set`) and reading it at exactly two points.

### 5.5 Event log

A timestamped, copyable log of everything the page observed: connection config, policy swaps,
per-request send / first-token / done / rejection events with real timings, full response text,
and burst summaries. It exists because an animated grid is illegible in a still frame and lossy
under GIF compression, whereas log lines survive both — and because "show me the actual
response" should not require opening devtools.

Capped at 600 lines. Autoscroll pins to the bottom only when the viewer is already there.

### 5.6 Comparison flow

**Compare policies** runs the identical burst under `fifo`, `priority`, then `wfq`, waiting for
the queue to drain between runs, and tabulates accepted / rejected / mean TTFT per priority
class / ratio. Priority classes are **1 and 5** and the ratio is **low ÷ high**, matching
`benchmarks/fairness.py` exactly, so a run here is directly comparable to the published numbers.
The original policy is restored when the run ends, including on failure.

The claim being demonstrated is the *ordering* — `fifo < 1.0 < wfq < priority` — not the
decimals, which move run to run. The page's caption and §7 say so.

## 6. Protocol contract the page depends on

Established by reading `tinyserve/api/app.py` and `schemas.py`:

- `POST /generate`, body `{prompt, max_tokens, stream: true, priority}`.
- On rejection: HTTP **503**, body `{"detail": {"reason": "..."}}`, header `Retry-After: 1`.
- On acceptance: `text/event-stream`, each frame literally `data: {token}\n\n`.
- **There is no terminal/`[DONE]` event.** Completion is signalled by the response stream
  closing. The page treats reader-done as completion.
- Because the server writes `data: ` with one space, and SSE strips exactly one leading space
  from a `data:` line, tokenizer-leading spaces survive correctly — the page must strip one
  space and no more, or every word will run together.

**Known limitation (server-side, pre-existing, not introduced here):** a token consisting of a
newline is written as `data: \n\n\n`, which is ambiguous under SSE framing — the newline is
consumed as the frame delimiter rather than delivered as content. Multi-line output therefore
renders flattened onto one line. This affects `curl -N` identically, so it is not a page bug.
Fixing it properly means JSON-encoding the SSE payload, which would change the wire format the
recording guide's curl commands depend on; out of scope here, noted for a future decision. The
page's default prompts are prose that does not depend on line breaks.

## 7. Honest limitations to state on the page itself

- **Browser connection limits.** Browsers cap concurrent connections per origin (~6 on HTTP/1.1).
  A 32-request burst does not put 32 requests on the wire simultaneously the way
  `benchmarks/burst.py` does — later ones queue *in the browser*, before ever reaching admission
  control. This inflates apparent TTFT and *understates* rejection counts. The page must say so
  in a footer rather than let a viewer read its numbers as benchmark-grade.
- **The KV rejection path is effectively unreachable from a browser at demo settings.** Measured
  at ~6 concurrent: `max_tokens` of 24–256 yields **zero** rejections, because each request
  reserves only `ceil((prompt + max_tokens)/16)` blocks — roughly 3 of 128 at `max_tokens=24`, so
  ~18/128 blocks are in use at full browser concurrency. `kv_cache_full` first appears at
  `max_tokens=400` (4 accepted / 2 rejected) and reaches 1/5 at `max_tokens=1024` — all of which
  mean 30-second generations, unusable in a recording. **Consequence:** demonstrating admission
  control from the page requires the *queue-depth* check (`TINYSERVE_MAX_QUEUE_DEPTH=3`), which
  fires realistically on small requests; the *KV-budget* check belongs to a terminal recording of
  `benchmarks/burst.py`, which has no connection cap. This is why the recording guide keeps the
  burst demo in a terminal rather than moving it here.
- **Browser-side timings.** TTFT here includes browser scheduling and paint; the server's own
  `ttft_seconds` histogram is the accurate one.
- Consequently: numbers on this page are *illustrative of behaviour*, and `docs/benchmarks.md`
  is cited on the page as the real measurement.

## 8. Acceptance criteria

1. `GET /demo` returns the page from a normally-started server; no CORS error in console.
0. `GET /config` reports live settings; `POST /config/policy` swaps policy and is reflected in
   the `scheduler_policy_decision_total{policy=...}` metric label; an unknown policy is a 400.
2. A 12-request burst at `n_seq_max=4` renders visibly interleaved token growth across cards —
   not sequential completion.
3. A burst large enough to exhaust budget renders rejected cards carrying the server's real
   `reason` string.
4. Under `mixed` priority with policy `wfq` or `priority`, high-priority cards reach first token
   before low-priority ones in the same burst.
5. The live strip's `queue_depth` visibly rises and falls during a burst.
6. Page is self-contained: no network request to any origin other than its own.
7. `ruff`, `mypy --strict`, and the existing test suite still pass.

## 9. Self-critique

- **This is a client dressed as a demo.** It proves the server does what it claims only insofar
  as a viewer trusts the page isn't faking it. Mitigation: it is served *by the runtime*, uses
  the documented public endpoint, and every command it issues has a curl equivalent in the
  recording guide. A sceptical viewer can reproduce any card with one curl.
- **It is the most "product-shaped" thing in a repo that is deliberately not a product**, and
  risks implying a UI is part of the system. Mitigation: it lives in `docs/` conceptually,
  behind a route named `/demo`, is absent from the architecture diagrams, and this PRD's §3
  states it is not a console.
- **Browser concurrency limits genuinely weaken the headline demo** (§7). The two-pane curl
  recording remains the more rigorous proof of interleaving; this page is the more *legible*
  one. Both ship; the README should present them that way rather than replacing one with the
  other.
