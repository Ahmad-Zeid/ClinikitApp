# CliniKit AI Assessment

Two exercises for the CliniKit AI Trainee assessment:

1. **Conversational clinic assistant** — understand a patient message, decide a safe action, reply.
2. **No-show prediction** — estimate whether a patient will miss an appointment.

AI tools were used while building this. Every decision below is one I can explain in an interview.

---

## Quick start

```bash
cd ClinikitApp
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add GROQ_API_KEY and/or GEMINI_API_KEY
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
.venv/bin/python ml/train.py
.venv/bin/python ml/predict.py
```

Notebook walkthrough: `ml/notebooks/part2_no_show.ipynb`

---

## Part 1 — approach

### The one idea

**The language model only suggests. Plain Python decides.**

```
patient message
  → UNDERSTAND   (Groq / Gemini / rules → structured Extraction)
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

Persistent sessions, real EHR calendar, human review queue for handoffs, continuous eval on Lebanese / Arabizi traffic, rate-limit budgets with paid tiers, audit log shipping.

### Layout

| Path | Role |
|------|------|
| `clinikit/agent/session.py` | One conversation; only place that writes memory |
| `clinikit/agent/policy.py` | Pure safety / routing |
| `clinikit/agent/tools.py` | Mocked clinic actions |
| `clinikit/agent/responder.py` | Hand-written replies |
| `clinikit/agent/phrasing.py` | Optional AI wording + fact check |
| `clinikit/agent/backends/` | Groq, Gemini, chain, rules |
| `web/index.html` | Chat + inspector |
| `eval/` | Hand-labelled test set + runner |

Results: `eval/RESULTS.md` (rules baseline). Live LLM numbers depend on today’s free-tier quota.

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

## Tests

```bash
.venv/bin/pytest -m "not live"
```

Live provider checks (uses API quota): `pytest -m live`

---

## Submission notes

- Secrets stay in `.env` (git-ignored). `.env.example` shows the keys.
- Free Groq / Gemini tiers run out; `auto` falls through the chain. Local usage is tracked in `.cache/daily_usage.json`.
- Part 1 is not “never wrong” — it is **safe by construction** on writes, with accuracy measured separately.
- Part 2 is an honest experiment on the supplied data. Its moderate score and data
  limits are documented rather than hidden.
