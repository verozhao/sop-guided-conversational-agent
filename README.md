# SOP-Guided Conversational Agent

An insurance claims support agent that follows a fixed business workflow and still talks like a person.

```
VERIFY_ID  ->  RESOLVE_INTENT  ->  PROCESS_CASE  ->  POST_PROCESS
 strict          guided             flexible          strict
```

The workflow is code. A language model reads what the caller said and words the reply. It never decides a phase change, never calls a tool, and never receives claim data before the caller is verified.

![Test console](docs/test-console.png)

## Quick start

### Docker

```bash
docker build -t sop-agent .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... sop-agent
# or: ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

Open http://localhost:8000.

### Python 3.10+

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-... .venv/bin/uvicorn server.app:app --port 8000
```

### The API token

The agent calls Claude through the Anthropic API. Give it a token in either way:

- set `ANTHROPIC_API_KEY` when you start the server, or
- paste the token into **Setup** in the UI. It stays in that browser tab's memory, travels in the `X-API-Key` header of your chat requests, and is never written to disk or to a log.

The default model is `claude-opus-5-5` at low effort. Each turn makes two model calls (understand, then phrase). Pick another model in the UI or with `SOP_MODEL`, for example `claude-haiku-5-5` for speed.

### No token

Everything also runs without a token. Leave the field empty and the agent uses its offline engine: rule-based understanding and template replies, driven by the same SOP policy. This is how the test suite runs. It handles common phrasings, which is narrower than what the model handles.

## Try it

The UI has a **Demo script** menu. Pick a script and press **Send next line** to step through it, or type freely.

| Script | What it shows |
|---|---|
| Demo test case | Identity extracted and verified, the "denied healthcare claim from January" hint remembered and used, grounded answer, email summary |
| Bonus: frustrated caller | Empathy first, the reason for verification, allowed options, no disclosure, then the saved question answered once verified |
| Bonus: refusal | Two rounds of persuasion, then a hand-off to a human with a context ticket |
| Out-of-scope questions | Polite decline, offer of a human on the third, transfer on the fourth |
| Representative caller | Listing check and policyholder consent as extra gates |
| Memory | A claim hint and an email preference given before verification, used later |
| Ambiguous claim | Disambiguation, then switching to a second claim |

A link such as `/?script=1&auto=1` plays a script on load. `&turns=2` stops after two lines.

The same scenarios run in a terminal, offline:

```bash
.venv/bin/python scripts/demo.py            # all scenarios
.venv/bin/python scripts/demo.py bonus      # one of: happy bonus refusal scope representative memory ambiguous
```

### Test data

| Policyholder | Policy | Date of birth | Phone | Email | ID last 4 | Claims |
|---|---|---|---|---|---|---|
| Margaret Chen | POL-9921 | 1985-03-15 | 650-521-2836 | margaret@email.com | 4472 | CL-2048 healthcare, denied (Jan 2026). CL-2011 healthcare, closed (Jan 2025). CL-1899 dental, closed (Nov 2025). CL-2102 auto, open (Feb 2026) |
| Ma Tian | POL-8836 | 1964-09-10 | 650-208-8799 | matian@example.com | 6688 | CL-3001 healthcare, denied (Mar 2026) |
| Ava Lopez | POL-1044 | 1990-08-21 | 650-388-2920 | ava.lopez@email.com | 9180 | none |
| Ya Wen Li | POL-7742 | 1989-12-03 | 650-521-2830 | yawen.li@gmail.com | 5317 | none |

David Chen is listed as Margaret Chen's authorized representative. The fixtures are dated early 2026, so the demo clock is fixed at 2026-03-05 (set `SOP_TODAY` to change it).

### The demo test case

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

| Expected | What happens | Where to see it |
|---|---|---|
| Extracts identity and verifies | Name, date of birth and SSN last four match one record. Three of five is the rule | Safety gates panel |
| No claim detail before verification | Until the gate opens, no claim tool is allowed and the reply writer receives no claim data | "Facts the reply was allowed to use" in the Last turn panel |
| Remembers the hint | Intent and "denied healthcare claim from January" are stored while still in VERIFY_ID | Memory panel, tagged with the phase and turn where it was heard |
| Uses the hint after verification | RESOLVE_INTENT filters Margaret's four claims down to CL-2048 without asking | Phase path `VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE`, and "Memory used" |
| Grounded answer | Denial reason, documents and appeal deadline come from tool results | Tools list and facts in the Last turn panel |

## How it works

### One turn

```
caller text
   |  full SSNs are redacted before anything is stored or sent
   v
1. UNDERSTAND   model with a JSON schema, or offline rules        -> Understanding
   v
2. DECIDE       SOP policy, plain code                            -> Directive
                memory, gates, tools, counters, transitions
   v
3. PHRASE       model writes the reply from the directive's facts (or the template is used)
   v
4. GUARD        code checks the reply and falls back to the template on any violation
   v
reply
```

The `Understanding` is a fixed structure: scope, caller role, identity values, intent, hints about which claim, emotion, yes or no, and so on. The `Directive` is what the SOP decided: a goal, the points to cover, the rules for this turn, and the complete set of facts the reply may use. It also carries a deterministic reply that satisfies it. That reply is used verbatim in offline mode and whenever the guard rejects what the model wrote.

### Who decides what

| Decision | Decided by |
|---|---|
| What the caller said: intent, identity values, emotion, scope | Model, constrained by a schema, then normalised in code |
| Whether identity is verified | Code. Three of five details must match one record |
| Phase changes | Code. A phase graph with entry gates |
| Which claim and which workflow path | Model proposes, code validates against the caller's own claims |
| Which tools run | Code. The path fixes the tools, and each tool is limited to certain phases |
| What facts a reply may contain | Code. Tool results for this turn only |
| Wording, empathy, answering the specific question | Model |
| Whether the reply is released | Code. The output guard |

### Different freedom in each phase

| Phase | The model is free to | Enforced in code |
|---|---|---|
| VERIFY_ID (strict) | Talk naturally, answer "why do you need this", handle partial answers, refusals and alternate identity fields | No claim tool is allowed in this phase, so no claim data exists in the turn. The gate opens in one function, on three matching details |
| RESOLVE_INTENT (guided) | Turn messy language into one of six intents and a choice of claim | The choice must be one of the verified caller's claims. Ambiguity leads to a question, a mismatch to a list of what is on file |
| PROCESS_CASE (flexible) | Answer follow-up questions in its own words, from the facts | The intent picks a path, the path picks the tools. Every claim number, amount and date in the reply must appear in the facts |
| POST_PROCESS (strict) | Word the offer and the goodbye | The email goes out only on an explicit choice, only to the address on file, and its content is composed in code |

Non-disclosure before verification does not depend on the model following an instruction. The data is not in the prompt. `tests/test_llm_path.py` asserts this on the exact text sent to the model.

### Memory across phases

Every turn fills memory slots whatever the current phase: identity values, role, intent, claim hints, the question asked, follow-up topics, the email preference, and free-form notes. A slot that belongs to a later phase waits there. The inspector shows where each slot was heard ("heard in VERIFY_ID, turn 1") and when it was used.

Examples: a caller who says "it's about my auto claim, and email me a summary at the end" during verification is taken straight to the auto claim once verified, and is later asked to confirm the summary they already requested.

### Scope

Each message is read as in scope, out of scope or mixed. Out-of-scope requests ("What is RL?") are declined politely and the caller is steered back to the current step. The phase does not change. On the third in a row the agent offers a human representative, and on the fourth it transfers. An in-scope message resets the count.

### Emotional support and SOP recovery (bonus)

- The understanding step labels each message as neutral, frustrated, angry, anxious, confused or sad. A non-neutral label puts tone guidance in the directive and the reply opens with one acknowledgement.
- Pushback on verification gets empathy, the reason for the step, credit for details already given, and the remaining options. Nothing is disclosed and the gate stays closed.
- A second pushback adds the offer of a human representative. A third stops the persuasion and transfers.
- "I don't have my SSN on me" is treated as a declined field. The agent offers the other fields.
- Giving details while upset counts as progress.
- A clear request for a human is honoured immediately in any phase. Several upset turns in a row after verification bring a proactive offer.
- Every transfer creates a ticket with what is known so far, so the caller does not repeat themselves. The ticket states whether identity was verified.

For the example in the brief ("I already told you who I am. This is ridiculous. Just tell me why my claim was denied."), the offline engine answers:

> I understand you're upset, and I'm sorry for the trouble this has caused you. You're right that you've given me your full name, thank you. Your claim file holds private health and financial information, so I have to confirm it's really you before I can share anything from it, including the reason for a decision. It only takes a moment. I need two more details to finish. You can choose from your date of birth, the phone number on your policy, the email address on your policy, or the last four digits of your SSN or national ID.

The question about the denial is saved and answered as soon as verification completes.

### Representative callers and consent

Someone calling for a policyholder has to pass three checks: the policyholder's details match, the caller is listed as an authorized representative, and the policyholder approves a consent request. The consent status comes from `consent_scenarios.json` (choose `default` or `timeout` in the UI before starting a conversation). Until consent is approved the session stays in VERIFY_ID.

### Output guard

A model reply is released only if:

- every claim number, policy number, email address, phone number, amount and date in it appears in the turn's facts or was said by the caller
- it contains no value held only in the records (identity numbers, other people's names, denial reasons, requested documents)
- it does not say the caller is verified, that something was sent, or that a transfer happened, unless the workflow reached that point

Otherwise the deterministic reply is sent and the rejection is shown in the inspector.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Token used when the browser does not send one |
| `SOP_MODEL` | `claude-opus-5-5` | Default model |
| `SOP_EFFORT` | `low` | Reasoning effort for both calls |
| `SOP_TODAY` | `2026-03-05` | Demo clock. A date, or `real` for the system date |
| `SOP_INSPECTOR` | `on` | `off` removes harness internals from API responses |
| `PORT` | `8000` | Port inside the container |

SOP limits live in `apps/insurance_claims/sop/common.py`: three failed identity attempts, two rounds of persuasion, a human offered at three out-of-scope messages and a transfer at four.

## Project layout

```
sop_harness/                 domain-agnostic harness
  core.py                    phase graph with gates, phase-gated tool registry, Directive, TurnTrace
  llm.py                     Anthropic client (structured output, refusal handling), scripted stand-in
apps/insurance_claims/
  agent.py                   the turn pipeline
  sop/workflow.py            phase graph, per-turn loop, cross-phase memory
  sop/verify_id.py           phase 1
  sop/resolve_intent.py      phase 2
  sop/process_case.py        phase 3 and the workflow paths
  sop/post_process.py        phase 4
  sop/common.py              limits, wording, hand-off, scope control
  verification.py            three-of-five identity matching
  tools.py                   claim, consent, email and hand-off tools
  understanding.py           the Understanding structure and its JSON schema
  nlu_llm.py, nlg.py         prompts for the two model calls
  nlu_rules.py               offline understanding
  guard.py                   output guard
  email_summary.py           summary composer
  fixtures/                  sample data
server/app.py                FastAPI app
server/static/index.html     test UI
scripts/demo.py              scripted conversations in the terminal
tests/                       168 tests, all offline
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

The suite covers the demo case, the verification gate, the bonus scenarios, scope control, claim resolution, the workflow paths, the email step, representative consent, harness enforcement (tool gating, illegal transitions), the output guard against leaking or overclaiming model replies, and the shape of the request sent to the Anthropic API.

## Status and limits

- No live model call was made during development, to avoid API charges. The model path is covered by scripted stand-ins and by request-shape tests against a fake client. A first run with a real token is the remaining check. If a model call fails for any reason, that turn falls back to the offline engine and the UI says so.
- The Docker image has not been built on the development machine.
- A safety classifier can decline a request on some models. For those models the client opts into the API's server-side fallback (`fallbacks: "default"`), and it drops the option on its own if the account does not have it.
- The offline engine is English only and rule-based.
- Sessions live in memory. Email delivery is simulated: the message appears in the Outbox panel.
