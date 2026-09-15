# CliniKit AI Assessment

Two exercises for the CliniKit AI Trainee assessment:

1. **Conversational clinic assistant** — understand a patient message, decide a safe action, reply.
2. **No-show prediction** — estimate whether a patient will miss an appointment.

AI tools were used while building this. Every decision below is one I can explain in an interview.

**Try Part 1 without installing anything: https://clinikit-app.vercel.app**

---

## Quick start

```bash
cd ClinikitApp
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add GEMINI_API_KEY (free, no card)
```

### Part 1 — chat assistant

```bash
# Terminal
.venv/bin/python -m clinikit.agent.cli --backend auto

# Web UI (chat + inspector)
.venv/bin/uvicorn clinikit.api.main:app --reload
# open http://127.0.0.1:8000
```

Optional: `CLINIKIT_NATURAL_REPLIES=0` turns off the second AI call that only softens wording.

### Part 2 — no-show model

```bash
.venv/bin/python ml/train.py          # the experiment
.venv/bin/python ml/ceiling_check.py  # could any other model do better? (no)
.venv/bin/python ml/predict.py        # example predictions
```

Notebook walkthrough: `ml/notebooks/part2_no_show.ipynb`
Full explanation from first principles: `ml/EXPLAINED.md`

---

## Part 1 — approach

### The one idea

**The language model only suggests. Plain Python decides.**

```
patient message
  → UNDERSTAND   (Gemini, with Groq as backup → structured Extraction)
  → POLICY       (decide what is allowed — G1–G7)
  → ACTION       (mocked clinic tools)
  → RESPONSE     (template, optionally reworded then fact-checked)
```

Why: models disagree on intent. On the brief’s own hedged example, two models picked different intents, but both set `is_hedged: true`. Safety hangs on that observable fact (“don’t book yet”), not on the contested label.

### Intent and extraction

- LLM backends return JSON shaped like `Extraction` (`clinikit/agent/schema.py`).
- Dates stay as phrases (`"tomorrow afternoon"`). Python in `temporal.py` turns them into real times.
- A keyword `rules` backend exists only as a baseline for eval — it is not used as a patient-facing fallback.

### Missing / ambiguous information

Policy asks instead of guessing: which doctor (two Khourys), which appointment, which numbered option, confirmation before any write.

### Avoiding incorrect automated actions

Writes (`create` / `cancel` / `reschedule`) only happen after an explicit yes against an offer **this system** made (`G2`). Hedged messages (`G1`) are look-ups only. Tests in `tests/test_safety_invariants.py` pin that.

### How to improve in production

Real EHR calendar behind the same `tools.py` method names, a human review queue for
handoffs, continuous evaluation on real Lebanese / Arabizi traffic, paid provider tiers
so quota is not a failure mode, and audit logs shipped somewhere durable.

Conversation memory is already handled: the server stores nothing, so it scales sideways
and survives restarts (`clinikit/agent/state.py`).

### Layout

| Path | Role |
|------|------|
| `clinikit/agent/session.py` | One conversation; only place that writes memory |
| `clinikit/agent/policy.py` | Pure safety / routing |
| `clinikit/agent/temporal.py` | Turns "tomorrow afternoon" into real times |
| `clinikit/agent/tools.py` | Mocked clinic actions |
| `clinikit/agent/responder.py` | Hand-written replies |
| `clinikit/agent/phrasing.py` | Optional AI wording + fact check |
| `clinikit/agent/state.py` | Signed conversation memory, so the server keeps none |
| `clinikit/agent/backends/` | Gemini, Groq, chain, rules |
| `clinikit/api/main.py` | HTTP service |
| `web/index.html` | Chat + inspector |
| `eval/` | Hand-labelled test set + runner |

Results on 55 hand-labelled cases (`eval/RESULTS.md`):

| backend | intent | hedge detection | doctor resolved | median latency | safety violations |
|---|---|---|---|---|---|
| `rules` (keyword baseline) | 48.0% | 98.2% | 98.2% | 0.00s | **0** |
| `gemini` | **98.0%** | 98-100% | **100%** | 1.05s | **0** |

The keyword baseline is there to answer one question: did the language model earn its
place? 48% to 98% says yes.

Two honest notes. The model is not perfectly repeatable, so a re-run moves a borderline
case or two — hedge detection sits at 98-100% depending on the run. And **safety
violations is not an accuracy score.** It counts times the appointment book changed on a
message containing no confirmation. It must be zero, and it is zero by construction
rather than by scoring well.

---

## Part 2 — approach

Part 2 uses the supplied `CliniKit_NoShow_Dataset.csv`: 3,000 historical
appointments, including 478 no-shows (15.9%).

The experiment keeps 20% of the rows hidden as a final test set. On the remaining
80%, five-fold cross-validation compares a no-skill baseline, logistic regression,
histogram gradient boosting, and random forest. This keeps model selection separate
from final evaluation.

### Model choice

`LogisticRegression` (scikit-learn) won the comparison:

- Best cross-validated ROC-AUC: **0.694**
- Best cross-validated PR-AUC: **0.368**
- Simpler to explain than the tree candidates
- Produces useful risk probabilities after categorical encoding and numeric scaling

The final hidden test result is **0.695 ROC-AUC** and **0.372 PR-AUC**. These are
moderate results, not production-ready claims.

### Why the score is moderate — and why that is the right answer

`ml/ceiling_check.py` exists to answer "could a better model do better?" It cannot:
every candidate lands between 0.67 and 0.70. More tellingly, **three of the ten columns
score 0.702 while all ten score 0.700** — the other seven add nothing measurable.

The limit is the data, not the model. Most of why someone misses an appointment — the car
would not start, the child got sick — was never recorded, and no algorithm recovers
information that was never collected. Published no-show models on real hospital data
typically reach 0.70–0.75, so this is roughly where the problem sits.

Full reasoning, written from first principles: `ml/EXPLAINED.md`.

### Metrics

No-shows are the minority class. **Accuracy alone is misleading.** We report:

- **ROC-AUC** — can we rank a true miss above a true attendance?
- **PR-AUC** — how precision and recall trade off for the rare no-show class
- **Precision / recall / F1 on no-shows** — what the chosen flagging point does
- **Brier score** — how close the predicted probabilities are to the outcomes

The decision threshold (**0.263**) maximises no-show F1 using training-only,
out-of-fold predictions. The hidden test set does not choose the threshold.

### What the model leans on

Permutation importance (shuffle one column, measure AUC drop) says the strongest
signal is `previous_no_shows`, followed by `reminder_sent`. This is association,
not proof that reminders cause attendance. See `ml/RESULTS.md` and
`ml/figures/feature_importance.png`.

### Product fit

Nightly job scores tomorrow’s list → “risk” badge in reception UI → extra SMS or waitlist offer. Advisory only; a person still decides. Retrain monthly on real outcomes.

---

## Deployment

Live at **https://clinikit-app.vercel.app**.

The interesting constraint: Vercel starts a short-lived worker per request and throws it
away, so the worker answering "yes" has never seen the offer being accepted. Rather than
add a database, the server keeps **no** memory at all — the conversation travels with the
patient and comes back with the next message.

It is signed (HMAC-SHA256), so a patient cannot forge an offer the clinic never made,
which is what `G2` depends on. `tests/test_state_roundtrip.py` runs that exact attack.

Deploying needs three environment variables: `GEMINI_API_KEY`, `CLINIKIT_STATE_SECRET`,
and `CLINIKIT_CACHE_DIR=/tmp/clinikit` (the disk is read-only elsewhere).

---

## Tests

```bash
.venv/bin/pytest          # 136 tests, about a second, no network
```

Live provider checks (uses API quota): `pytest -m live`
Re-run the agent evaluation: `.venv/bin/python eval/run_eval.py --backends rules,gemini`

The suite covers the seven safety guarantees, conversation memory, date and time
handling, the reply fact-checker, and the things that only break once deployed
(`tests/test_deployable.py`).

---

## Submission notes

- Secrets stay in `.env` (git-ignored). `.env.example` shows the keys.
- Free Gemini / Groq tiers run out; `auto` falls through the chain, Gemini first. Usage is
  tracked in `.cache/daily_usage.json` so a restart does not re-discover it the expensive way.
- Part 1 is not “never wrong” — it is **safe by construction** on writes, with accuracy measured separately.
- Part 2 is an honest experiment on the supplied data. Its moderate score and data
  limits are documented rather than hidden.
