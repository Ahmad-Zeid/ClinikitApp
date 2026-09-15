# Part 2, explained from scratch

This file is for me. It explains what the no-show model does, why each decision was
made, and what I would say if asked about any of it. Nothing here assumes prior
knowledge.

---

## 1. What we were asked to do

A clinic has a list of past appointments. For each one it recorded a few facts — the
patient's age, how far ahead they booked, whether a reminder went out, how many times
they had missed before — and whether the patient actually turned up.

We want to look at *tomorrow's* appointments and guess which ones will be missed, so the
clinic can do something about it: send another reminder, phone them, or offer the slot
to someone on the waiting list.

That is the whole task. Everything below is detail.

---

## 2. The data

3,000 appointments. **478 of them were missed — 15.9%.**

That imbalance is the single most important fact about this problem, and I'll come back
to it twice.

Ten columns describe each appointment:

| column | what it is |
|---|---|
| `age` | patient's age |
| `gender` | Male / Female |
| `appointment_type` | New Consultation, Follow-up, Routine Check, Procedure, Urgent Visit |
| `days_before_appointment` | how far ahead they booked |
| `previous_appointments` | how many they have had before |
| `previous_no_shows` | how many they have missed before |
| `weekday` | Monday … Saturday |
| `appointment_time` | a two-hour band, e.g. 16:00-18:00 |
| `reminder_sent` | 0 or 1 |
| `new_patient` | 0 or 1 |

There is an eleventh column, `appointment_id`. **We throw it away.** It is just a row
number. A model given row numbers can memorise the spreadsheet — "row 412 was a no-show"
— which looks like learning and is useless on a patient it has never seen.

No missing values, no duplicate rows. I checked, and `ml/train.py` re-checks every run
and refuses to train if the file is broken.

---

## 3. The one trap in this problem

**84% of patients turn up.** So I can write a "model" in one line:

> Predict that everybody attends.

That model is **84% accurate** and completely worthless. It never flags anyone, so the
clinic does nothing differently, and all 478 missed appointments still happen.

This is why *accuracy is the wrong measure here*, and being able to say that sentence is
most of what this exercise is testing. Whenever one answer is much more common than the
other, accuracy rewards you for ignoring the rare case — and the rare case is the entire
point.

So we measure other things instead.

### The measures we do use

**Recall** — of all the patients who actually missed, how many did we flag?
*Miss this and no-shows slip through.*

**Precision** — of all the patients we flagged, how many actually missed?
*Miss this and we pester people who were always going to turn up.*

These two fight each other. Flag everybody and recall is perfect, precision is terrible.
Flag nobody and the reverse. The job is choosing where to sit between them.

**ROC-AUC** — take one patient who missed and one who attended, at random. How often does
the model give the higher risk score to the one who missed? 0.5 means it is guessing.
1.0 means perfect. It's useful because it doesn't depend on where we draw the line.

**PR-AUC** — a summary of the precision/recall trade-off across every possible line.
Better suited to a rare outcome than ROC-AUC is.

**Brier score** — are the *probabilities* honest? If the model says "30% risk" about a
hundred patients, roughly thirty of them should miss. Lower is better. Ours is 0.121.

---

## 4. How the experiment is set up

This is the part I'd want to get right in front of an interviewer, because it's the part
that makes the final number trustworthy.

**Step one: hide 20% of the rows.** 600 appointments are locked away and not looked at.

**Step two: on the other 2,400, run five-fold cross-validation.** That means: split those
2,400 into five equal piles. Train on four piles, test on the fifth. Repeat five times so
every pile gets a turn as the test. Average the five scores.

Why bother? Because a single train/test split can be lucky. Five rounds and an average is
harder to fluke.

**Step three: pick the winning model, using only those cross-validation scores.**

**Step four: only now, look at the 600 hidden rows. Once.**

The reason for all this ceremony: if I look at the hidden rows while choosing the model,
I am choosing the model that happens to suit those particular rows. The final score then
describes my luck rather than the model's skill. It's reading the answer key before
sitting the exam.

**The decision threshold is chosen the same careful way** — from the training rows only
(see §6). The hidden 600 never influence anything; they only ever get scored.

---

## 5. Which model, and why

Four candidates, compared fairly:

| model | cross-validated ROC-AUC |
|---|---|
| **logistic regression** | **0.694** |
| random forest | 0.672 |
| gradient boosting | 0.648 |
| always-guess-the-average (baseline) | 0.500 |

**Logistic regression won.** It also happens to be the easiest to explain, which is a
real advantage and not a consolation prize.

### What logistic regression actually is

Plain version: it gives every fact a weight — a number saying how much that fact pushes
towards "will miss" or "will turn up" — adds them all up, and squashes the total into a
percentage between 0 and 100.

That's it. It's a weighted scorecard.

> *Started with 0. Two previous no-shows? Add a lot. Reminder was sent? Subtract some.
> It's a Procedure? Subtract some more. Total the score, convert to a percentage.*

Because it's just weights, I can read them straight out and see exactly what the model
believes. With the tree-based models I'd have to interrogate them. Here the explanation
*is* the model.

### The baseline row matters

`dummy_baseline` scores exactly 0.500 — pure coin-toss. Including it proves the other
models learned something real. A comparison without a baseline can't tell you whether
0.694 is good or whether the task is trivial.

---

## 6. Choosing where to draw the line

The model outputs a probability, like 0.31. To act, the clinic needs a yes or no. So we
need a cut-off.

The obvious choice is 0.5 — flag anyone above 50%. That's wrong here. Because only 16% of
patients miss, almost nobody ever reaches 50%, so a 0.5 cut-off flags almost no one and
we're back to the useless model.

So the cut-off is **0.263**, chosen by finding the point that gives the best balance of
precision and recall (the best "F1", which is just a way of averaging the two).

**Crucially, that number was worked out using only the training rows.** Tuning the
threshold on the hidden test set would be the same cheat as picking the model on it.

---

## 7. The results

On the 600 hidden appointments:

| | |
|---|---|
| ROC-AUC | **0.695** |
| PR-AUC | **0.372** (a coin toss would score 0.16) |
| Precision on no-shows | 0.384 |
| Recall on no-shows | 0.344 |
| Brier score | 0.121 |
| Accuracy | 0.807 |

Note that accuracy of 0.807 is *lower* than the do-nothing model's 0.84 — and the model
is still far more useful, because it actually catches people. That contrast is worth
pointing at.

In plain terms:

> Of 96 patients who really did miss their appointment, the model flagged **33**.
> It also flagged **53** people who turned up anyway.

So: it catches about a third of no-shows, and about two in five of its warnings are
right.

**Is that good?** It is modest, and I'd say so plainly. But compare it to the clinic's
alternative, which is knowing nothing. Thirty-three prevented no-shows is thirty-three
slots recovered, and the cost of a wrong flag is one extra text message. For that cost,
a third is worth having.

---

## 8. The question I expect to be asked: "why so low?"

This is the part I'd most want to get right, so I checked it properly rather than
guessing. `ml/ceiling_check.py` runs the check and prints the evidence.

**Finding one — no model does better.** Everything lands in the same place:

```
logistic regression         0.700
logistic, regularised       0.703
logistic, balanced          0.701
random forest               0.699
gradient boosting, tuned    0.694
logistic + interactions     0.675
```

**Finding two — and this is the striking one — three columns beat all ten:**

```
previous_no_shows alone     0.669
+ reminder_sent             0.689
+ previous_appointments     0.702   <- three columns
all ten columns             0.700   <- everything
```

The other seven columns contribute **nothing measurable**. Age, gender, weekday,
time-of-day, appointment type — all of it, together, is worth less than noise.

**So the limit is the data, not the model.** The file records what the clinic happened to
write down. But most of why somebody misses an appointment — the car wouldn't start, the
child got sick, the bus never came, they simply forgot — was never written down anywhere,
and no algorithm can recover information that was never collected.

That's the honest answer, and it's a better answer than a higher number. Chasing 0.71
with a stack of tricks would show I didn't understand the problem.

**For context**, published no-show models trained on real hospital records usually land
around 0.70–0.75. We are not far off, on 3,000 synthetic rows.

---

## 9. What the model leans on

Two ways of asking, and they agree.

**Permutation importance** — shuffle one column at random, see how much the score drops.
A big drop means the model was relying on it.

```
previous_no_shows        0.199   <- dominant
reminder_sent            0.024
appointment_type         0.008
days_before_appointment  0.007
age                      0.004
...everything else       ~0
weekday                 -0.019   <- negative: noise
```

**The weights themselves** point the same way: `previous_no_shows` is the largest positive
weight (more past misses → higher risk), `reminder_sent` is negative (reminder → lower
risk).

A negative importance is not "negatively important" — it means shuffling that column
*helped* slightly, which is what noise looks like. Worth saying rather than hiding.

### The sentence I must be careful with

The data shows patients who got a reminder missed less often — 14.2% versus 22.2%.

It is **very** tempting to say "reminders reduce no-shows by 8 points". I should not say
that, and if asked I should explain why:

> We didn't *run an experiment*. Nobody flipped a coin to decide who got a reminder. The
> clinic decided — and it may well have sent reminders to the patients it could most
> easily reach, who are also the patients most likely to turn up anyway. The reminder and
> the attendance may share a cause rather than one causing the other.

**A pattern in old data is an association. Proving cause needs a trial** where reminders
are assigned at random. That's a real thing a clinic could run, and it's a good answer to
"how would you improve this".

---

## 10. Putting it in a product

A nightly job scores tomorrow's appointments. Reception opens the morning list and sees a
risk badge next to a few names. They decide what to do — another text, a phone call, or
offering the slot to the waiting list.

Three rules I'd insist on:

1. **The model advises; a person decides.** It must never cancel, overbook, or deny care
   on its own. The cost of being wrong lands on a patient who did nothing wrong.
2. **The clinic picks the threshold, not me.** Where to sit between precision and recall
   is a business judgement — how much is an empty slot worth, how annoying is a needless
   text — and they're the ones who know.
3. **Watch it over time.** Retrain monthly on real outcomes, check it hasn't drifted, and
   check it isn't performing worse for any particular group of patients.

---

## 11. Limits I would state before being asked

- **3,000 rows, 478 no-shows.** Small. The ±0.03 spread across cross-validation folds
  shows how much wobble that leaves.
- **No patient identifier.** If one person appears three times, their rows can land on
  both sides of the train/test split, which can flatter the score slightly. With an ID I
  would split by patient, not by appointment.
- **Associations, not causes.** Especially the reminder finding.
- **Synthetic-looking data.** The clean 3,000 rows with no missing values don't behave
  like a real clinic export. Real data would be messier and probably harder.
- **Modest performance, honestly reported.** 0.695, at what looks like the ceiling of what
  this file can support.

---

## 12. Where everything lives

| file | what it does |
|---|---|
| `ml/train.py` | the whole experiment: load, check, compare, pick, evaluate, save |
| `ml/ceiling_check.py` | the "could any model do better?" evidence in §8 |
| `ml/predict.py` | loads the saved model and scores three example patients |
| `ml/RESULTS.md` | generated results — never edited by hand |
| `ml/models/metrics.json` | every number, machine-readable |
| `ml/notebooks/part2_no_show.ipynb` | the walkthrough version |

Reproduce everything:

```bash
.venv/bin/python ml/train.py
.venv/bin/python ml/ceiling_check.py
.venv/bin/python ml/predict.py
```

The random seed is fixed at 42, so the numbers come out the same every time.
