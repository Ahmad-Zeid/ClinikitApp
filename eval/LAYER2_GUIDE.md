# Layer 2 — the hard messages (Ahmad writes these)

## What this is for

The test set currently has 20 cases: the brief's 9 examples, and 10 attacks. Right now the
keyword reader scores **100%** on them.

That number is fake. I wrote the keyword rules while reading the brief's examples, so of
course it passes them. A test set is only worth something when it contains messages nobody
designed the code around.

That is what these 30 are. They are the ones that will actually separate the keyword
reader from Gemini, and the gap between the two is the evidence that using an LLM was the
right call.

## Why you write them and not me

You know how people in Lebanon actually text a clinic. I can write a plausible imitation,
but plausible imitations are exactly what a test set must not contain -- they end up
testing the messages I was already able to imagine.

## What to write

Open `eval/layer2_messages.txt`. Write one message per line under each heading. Do not
write the answers -- we will do the labelling together afterwards, and me labelling your
messages is fine. What must not happen is a model writing both the message and the answer.

Target: **30 messages**, roughly split like this.

| Category | How many | What I mean |
|---|---|---|
| Arabizi / Lebanese | 6 | Arabic in Latin letters and numbers. "3anjad", "badde", "bukra", "shu". Mixed Arabic-English is ideal -- that is how people really write. |
| Typos and sloppy typing | 5 | Missing letters, no punctuation, autocorrect damage, "apointmnt", "tmrw", all lowercase. |
| Missing information | 4 | A real request that leaves something out. "I need to see someone next week" -- which doctor? |
| Two things at once | 3 | "cancel friday and book me monday instead" -- one message, two requests. |
| Vague or hesitant | 4 | Not a clear instruction. "maybe I should come in", "I was thinking about...". Some should be genuine hedges, some just polite. |
| A symptom, not a request | 3 | "my back has been hurting since Saturday". They never actually ask for anything. |
| Annoyed or blunt | 2 | Short, irritated, no pleasantries. Real patients are sometimes in a bad mood. |
| Changing mind mid-message | 3 | "is dr karim free thursday? actually no, make it friday" |

## Rules

- **Write them as a real patient would.** If it looks tidy, it is probably too easy.
- **Do not look at the code while writing.** Write the message you would actually send.
- **Hard is good.** A message you are not sure the agent can handle is the most valuable
  kind. We want to find failures, not avoid them.
- **Include a few you think will break it.** Those are the best ones.

## What happens next

You write the messages. We go through them together and agree the correct answer for each
-- intent, doctor, whether it is a genuine hedge. Then they join the test set and we
re-run the evaluation and see the real numbers.

Expect the keyword reader to drop a long way. That is the point.
