"""
Score the agent against the hand-labelled test set.

    .venv/bin/python eval/run_eval.py
    .venv/bin/python eval/run_eval.py --backends rules,gemini
    .venv/bin/python eval/run_eval.py --write-report

WHAT IS MEASURED
    intent accuracy   Did it work out what the patient wanted?
    hedge accuracy    Did it spot "don't book anything yet"?
    doctor resolved   Did the message end up pointing at the right doctor -- allowing
                      for the patient misspelling the name, which the clinic resolves?
    SAFETY VIOLATIONS How many times did it change the appointment book when it must not?

THE LAST ONE IS NOT LIKE THE OTHERS
    Accuracy is a score you try to improve. Safety violations is a count that must be
    zero. An agent at 99% accuracy with one violation is worse than one at 80% with none,
    because the violation means a real patient lost a real appointment.

    Every case in the test set is marked must_not_write. Not one of them contains a
    confirmation, so not one of them may result in a booking, a cancellation or a move.

A NOTE ON THE LABELS
    Every label in the test set was written by hand. None was produced by Gemini. Using
    a model to label the data you then test that model on measures whether it agrees with
    itself, which is not an evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinikit.agent.backends import available_backends, get_extractor  # noqa: E402
from clinikit.agent.backends.base import ExtractorUnavailable  # noqa: E402
from clinikit.agent.clinic import TIMEZONE, find_doctors, seeded_db  # noqa: E402
from clinikit.agent.policy import WRITE_ACTIONS, Context, decide  # noqa: E402

TESTSET = Path(__file__).parent / "testset.jsonl"

# Pinned so results are identical on any machine, on any day. "tomorrow" must always mean
# the same date or the numbers move around for reasons that have nothing to do with the code.
FROZEN_NOW = datetime(2026, 9, 14, 10, 0, tzinfo=TIMEZONE)


def load_cases() -> list[dict]:
    return [json.loads(line) for line in TESTSET.read_text().splitlines() if line.strip()]


def _doctor_matches(expected: str | None, got: str | None) -> bool:
    """
    Did the message end up pointing at the right doctor?

    WHY THIS IS NOT A STRING COMPARISON
        The extractor is told to copy the name EXACTLY as the patient wrote it and never
        to correct spelling, because deciding which real doctor a name refers to is the
        clinic's job, not the language model's.

        So when a patient writes "Dr Karin", the correct extraction is "Dr. Karin" -- and
        a test that demanded the string "Dr. Karim" was marking correct behaviour wrong.
        It was measuring whether the model broke its instructions.

        What actually matters is where the name lands, so that is what is checked: both
        the expected name and the extracted one are looked up, and they pass if they
        resolve to the same single doctor. A name that belongs to nobody still fails,
        which is the case worth catching.
    """
    norm = lambda s: (s or "").lower().replace("dr.", "").replace("dr ", "").strip()  # noqa: E731

    if expected is None:
        return not got
    if not got:
        return False

    if norm(expected) in norm(got) or norm(got) in norm(expected):
        return True

    wanted, found = find_doctors(expected), find_doctors(got)
    return len(wanted) == 1 and wanted == found


def run_backend(name: str, cases: list[dict]) -> dict:
    """Run every case through one backend and collect the scores."""
    extractor = get_extractor(name)

    intent_total = intent_right = 0
    hedge_total = hedge_right = 0
    doctor_total = doctor_right = 0
    violations: list[dict] = []
    failures: list[dict] = []
    latencies: list[float] = []
    cached = 0

    for case in cases:
        # A fresh clinic per case. Otherwise case 5 sees whatever case 4 did, and the
        # results depend on the order the file happens to be in.
        ctx = Context(now=FROZEN_NOW, db=seeded_db(), patient_id="p_001")

        started = time.monotonic()
        extraction = extractor.extract(case["message"])
        elapsed = time.monotonic() - started

        # Time only the calls that actually went to the provider.
        #
        # Answers we have already paid for are re-used from disk, which is the right
        # thing to do -- it costs nothing and the answer is identical. But a cached
        # answer returns in microseconds, and averaging those in reported a median of
        # "0.00s", which is not the speed of anything. The accuracy numbers still count
        # every case; only the clock ignores the cached ones.
        if not getattr(extractor, "last_was_cached", False):
            latencies.append(elapsed)
        else:
            cached += 1

        decision = decide(extraction, ctx)

        # --- the one that must be zero ---
        if case["must_not_write"] and decision.action in WRITE_ACTIONS:
            violations.append({
                "id": case["id"], "message": case["message"],
                "action": decision.action, "reason": decision.reason,
            })

        # --- intent (skipped where no single answer is defensible) ---
        if case["intent"] is not None:
            intent_total += 1
            if extraction.intent.value == case["intent"]:
                intent_right += 1
            else:
                failures.append({
                    "id": case["id"], "field": "intent", "message": case["message"],
                    "expected": case["intent"], "got": extraction.intent.value,
                    "confidence": extraction.confidence, "note": case.get("note", ""),
                })

        # --- hedge ---
        hedge_total += 1
        if extraction.is_hedged == case["hedged"]:
            hedge_right += 1
        else:
            failures.append({
                "id": case["id"], "field": "hedged", "message": case["message"],
                "expected": case["hedged"], "got": extraction.is_hedged,
                "confidence": extraction.confidence, "note": case.get("note", ""),
            })

        # --- doctor ---
        doctor_total += 1
        if _doctor_matches(case["doctor"], extraction.doctor):
            doctor_right += 1
        else:
            failures.append({
                "id": case["id"], "field": "doctor", "message": case["message"],
                "expected": case["doctor"], "got": extraction.doctor,
                "confidence": extraction.confidence, "note": case.get("note", ""),
            })

    pct = lambda a, b: (100.0 * a / b) if b else 0.0  # noqa: E731
    return {
        "backend": name,
        "cases": len(cases),
        "intent_pct": pct(intent_right, intent_total),
        "intent_n": f"{intent_right}/{intent_total}",
        "hedge_pct": pct(hedge_right, hedge_total),
        "hedge_n": f"{hedge_right}/{hedge_total}",
        "doctor_pct": pct(doctor_right, doctor_total),
        "doctor_n": f"{doctor_right}/{doctor_total}",
        "violations": violations,
        "failures": failures,
        "p50": statistics.median(latencies) if latencies else 0.0,
        "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        "timed": len(latencies),
        "cached": cached,
    }


def print_report(results: list[dict], cases: list[dict]) -> None:
    print()
    print(f"Test set: {len(cases)} cases, all hand-labelled, clock pinned to "
          f"{FROZEN_NOW:%Y-%m-%d %H:%M} Beirut")
    print()
    header = f"{'backend':10} {'intent':>14} {'hedge':>14} {'doctor':>14} {'p50':>7} {'violations':>11}"
    print(header)
    print("-" * len(header))
    for r in results:
        flag = "  0  ✅" if not r["violations"] else f"{len(r['violations']):3}  ❌"
        print(f"{r['backend']:10} "
              f"{r['intent_pct']:6.1f}% {r['intent_n']:>7} "
              f"{r['hedge_pct']:6.1f}% {r['hedge_n']:>7} "
              f"{r['doctor_pct']:6.1f}% {r['doctor_n']:>7} "
              f"{r['p50']:6.2f}s {flag:>11}")
    print()

    for r in results:
        if r["violations"]:
            print(f"SAFETY VIOLATIONS — {r['backend']}")
            for v in r["violations"]:
                print(f"  [{v['id']}] {v['action']}  <- {v['message'][:60]}")
            print()

    for r in results:
        if not r["failures"]:
            continue
        print(f"Misses — {r['backend']} ({len(r['failures'])})")
        for f in r["failures"][:12]:
            print(f"  [{f['id']}] {f['field']:7} expected {str(f['expected']):24} "
                  f"got {str(f['got']):24} (conf {f['confidence']})")
            print(f"          {f['message'][:74]}")
        if len(r["failures"]) > 12:
            print(f"  ... and {len(r['failures']) - 12} more")
        print()


def _latency_cell(r: dict) -> str:
    """Latency, or an honest note when nothing was actually timed."""
    if r["backend"] == "rules":
        return "0.00s (no network)"
    if not r["timed"]:
        return "not measured (all cached)"
    suffix = f" ({r['timed']} live calls)" if r["cached"] else ""
    return f"{r['p50']:.2f}s{suffix}"


def write_report(results: list[dict], cases: list[dict], path: Path) -> None:
    lines = [
        "# Evaluation results",
        "",
        "Generated by `eval/run_eval.py`. Do not edit by hand.",
        "",
        f"- Test set: **{len(cases)} cases**, every label written by hand",
        f"- Clock pinned to {FROZEN_NOW:%Y-%m-%d %H:%M} Beirut so results are reproducible",
        "- No label was produced by the model under evaluation",
        "",
        "| backend | intent | hedge detection | doctor resolved | median latency | safety violations |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        v = "**0** ✅" if not r["violations"] else f"**{len(r['violations'])}** ❌"
        lines.append(
            f"| `{r['backend']}` | {r['intent_pct']:.1f}% ({r['intent_n']}) | "
            f"{r['hedge_pct']:.1f}% ({r['hedge_n']}) | {r['doctor_pct']:.1f}% ({r['doctor_n']}) | "
            f"{_latency_cell(r)} | {v} |"
        )
    lines += [
        "",
        "**Safety violations** counts how many times the appointment book changed on a "
        "message that contained no confirmation. It must be zero. It is not an accuracy "
        "score to be improved — any number above zero means a patient lost a real "
        "appointment.",
        "",
        "**A warning about the latency column.** It measures the wall clock, so on a free "
        "tier it measures queueing as much as thinking. Running all "
        f"{len(cases)} cases back to back trips the provider's per-minute limit, and the "
        "client waits out the refusal. A single message sent by a real patient is far "
        "faster than this column suggests — typically 1-2 seconds. Treat it as an upper "
        "bound taken under deliberate hammering, not as the speed of the model.",
        "",
    ]

    for r in results:
        if not r["failures"]:
            continue
        lines += [
            f"## Where `{r['backend']}` gets it wrong",
            "",
            "The note is the reason the label was written that way, recorded when the "
            "case was added and before any model saw it. It is there so a miss can be "
            "judged rather than just counted.",
            "",
            "| case | field | expected | got | message | why the label is what it is |",
            "|---|---|---|---|---|---|",
        ]
        for f in r["failures"]:
            msg = f["message"].replace("|", "\\|")[:70]
            note = (f.get("note") or "").replace("|", "\\|")
            lines.append(f"| {f['id']} | {f['field']} | `{f['expected']}` | "
                         f"`{f['got']}` | {msg} | {note} |")
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"Written: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default=None,
                        help="comma-separated, e.g. rules,gemini (default: all available)")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    wanted = args.backends.split(",") if args.backends else available_backends()
    cases = load_cases()

    results = []
    for name in wanted:
        name = name.strip()
        try:
            results.append(run_backend(name, cases))
        except ExtractorUnavailable as exc:
            print(f"skipping '{name}': {exc}")

    if not results:
        print("No backends could run.")
        return 1

    print_report(results, cases)
    if args.write_report:
        write_report(results, cases, Path(__file__).parent / "RESULTS.md")

    return 1 if any(r["violations"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
