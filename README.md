# SOP-Guided Conversational Agent

An insurance claims support agent that follows a fixed business workflow and still talks like a person.

```
VERIFY_ID  ->  RESOLVE_INTENT  ->  PROCESS_CASE  ->  POST_PROCESS
 strict          guided             flexible          strict
```

The workflow is code. A language model reads what the caller said and words the reply. It never decides a phase change, never calls a tool, and never receives claim data before the caller is verified.

![Test console](docs/test-console.png)

## Quick start

```bash
git clone <this repository> && cd sop-guided-conversational-agent
```

### Docker

```bash
docker build -t sop-agent .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... sop-agent
# or: ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

Open http://localhost:8000.

### Python 3.11 or newer (tested on 3.12)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-... .venv/bin/uvicorn server.app:app --port 8000
```

Open http://localhost:8000. Leave `ANTHROPIC_API_KEY` out to start without a token.

### The API token

The agent calls Claude through the Anthropic API. Give it a token in any of three ways:

- set `ANTHROPIC_API_KEY` when you start the server
- copy `.env.example` to `.env` and put the token there. The server and `docker compose` both read it, and `.env` is ignored by git
- paste the token into **Setup** in the UI. It stays in that browser tab's memory, travels in the `X-API-Key` header of your chat requests, and is never written to disk or to a log

The default model is `claude-opus-5-5` at low effort. Each turn makes two model calls (understand, then phrase), which takes about 5 seconds and costs about 2 cents. Pick another model in the UI or with `SOP_MODEL`. `claude-haiku-5-5` answers in about 2 seconds for a small fraction of a cent.

### No token

Everything also runs without a token. Leave the field empty and the agent uses its offline engine: rule-based understanding and template replies, driven by the same SOP policy. This is how the test suite runs. It handles common phrasings, which is narrower than what the model handles. It is built to fail safe where a rule can decide it: an email, a transfer, a change of claim and a counted identity attempt are read from clear wording, mixed or unclear wording is asked about again, and when a caller mentions somebody else without saying who is calling, the agent asks before it opens the gate. It is still a rule engine, so unusual phrasings can be misread.

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

In three of the scripts the caller does not say who they are, so the agent asks "are you the policyholder yourself?" once the details match, and the script answers yes. The demo test case says "I'm the policyholder" in its first message and is not asked.

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
| Extracts identity and verifies | Name, date of birth and SSN last four match one record. Three of five is the rule. The policy number helps find the record and does not count as one of the three. The message also says "I'm the policyholder", so who is calling is settled in the same turn | Safety gates panel |
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
| Whether identity is verified | Code. Three of five details must match one record. A wrong value is a failed attempt and is discarded, and three failed attempts hand the caller to a person |
| Phase changes | Code. A phase graph with entry gates |
| Which claim and which workflow path | Model proposes, code validates against the caller's own claims |
| Which tools run | Code. The path fixes the tools, and each tool is limited to certain phases |
| What facts a reply may contain | Code. Tool results for this turn only |
| Wording, empathy, answering the specific question | Model |
| Whether the reply is released | Code. The output guard |

### Different freedom in each phase

| Phase | The model is free to | Enforced in code |
|---|---|---|
| VERIFY_ID (strict) | Talk naturally, answer "why do you need this", "which three?" or "can I use my address instead?", handle partial answers, refusals and alternate identity fields | No claim tool is allowed in this phase, so no claim data exists in the turn. The gate opens in one function, on three matching details |
| RESOLVE_INTENT (guided) | Turn messy language into one of six intents and a choice of claim | The choice must be one of the verified caller's claims. Ambiguity leads to a question, a mismatch to a list of what is on file |
| PROCESS_CASE (flexible) | Answer follow-up questions in its own words, from the facts | The intent picks a path, the path picks the tools. A message that asks two things takes both paths. Every claim number, amount and date in the reply must appear in the facts. A request no path covers gets an offer of a human representative |
| POST_PROCESS (strict) | Word the offer and the goodbye | The email goes out only on an explicit choice, only to the address on file, and its content is composed in code |

Non-disclosure before verification does not depend on the model following an instruction. The data is not in the prompt. `tests/test_llm_path.py` asserts this on the exact text sent to the model.

### Memory across phases

Every turn fills memory slots whatever the current phase: identity values, role, intent, claim hints, the question asked, follow-up topics, the email preference, and free-form notes. A slot that belongs to a later phase waits there. The inspector shows where each slot was heard ("heard in VERIFY_ID, turn 1") and when it was used.

Examples: a caller who says "it's about my auto claim, and email me a summary at the end" during verification is taken straight to the auto claim once verified, and is later asked to confirm the summary they already requested. A second question asked while the first is still waiting is added to it, and both are answered after verification.

### Scope

Each message is read as in scope, out of scope or mixed. Out-of-scope requests ("What is RL?") are declined politely, in fixed wording, and the caller is steered back to the current step. The phase does not change. When the count reaches three the agent offers a human representative, and at four it transfers. A relevant message takes one off the count. It does not reset it, so slipping one relevant line between unrelated questions does not start the count again.

A question about the step itself is in scope: before verification, "what do you mean?" or "what happens if I get it wrong?" is answered and never counted.

### Emotional support and SOP recovery (bonus)

- The understanding step labels each message as neutral, frustrated, angry, anxious, confused or sad. A non-neutral label puts tone guidance in the directive and the reply opens with one acknowledgement.
- Pushback on verification gets empathy, the reason for the step, credit for details already given, and the remaining options. Nothing is disclosed and the gate stays closed.
- A second pushback adds the offer of a human representative. A third stops the persuasion and transfers.
- "I don't have my SSN on me" is treated as a declined field. The agent offers the other fields.
- Giving details while upset counts as progress.
- A clear request for a human is honoured immediately in any phase. Mentioning a person ("was this decided by an actual person?") or saying "don't transfer me" is not a request.
- After verification, a confused caller who asks nothing specific gets the explanation again and is asked which part to go over. An upset caller who asks nothing is asked what was missed, and after three such turns is asked whether they would like a person.
- Every transfer creates a ticket with what is known so far, so the caller does not repeat themselves. The ticket states whether identity was verified.

For the example in the brief ("I already told you who I am. This is ridiculous. Just tell me why my claim was denied."), said by a caller who has so far given only a name, the offline engine answers:

> I understand you're upset, and I'm sorry for the trouble this has caused you. You're right that you've given me your full name, thank you. Your claim file holds private health and financial information, so I have to confirm it's really you before I can share anything from it, including the reason for a decision. It only takes a moment. I need two more details to finish. You can choose from your date of birth, the phone number on your policy, the email address on your policy, or the last four digits of your SSN or national ID.

The question about the denial is saved and answered as soon as verification completes.

### Who is calling

Three matching details say whose policy it is. They do not say who is calling, and a relative often knows them. So once the details match, the agent asks "are you the policyholder yourself?" before anything is shared. The question is skipped only when the caller has already said so outright ("I'm the policyholder", "it's my own policy") in a message that mentions nobody else, as the demo test case does.

The answer is read from the caller's own words, whichever engine understood the message:

| The caller | Examples | What happens |
|---|---|---|
| says yes plainly, or states outright that the policy is theirs | "Yes", "Yes, I am. Why was it denied?", "That's me", "Margaret Chen" | The gate opens |
| says no, gives another name as their own, speaks of the policyholder as somebody else, or describes themselves as that person's relative or helper | "No", "I'm David Chen", "it's my mom's policy", "Yes, I'm her son", "I work for Margaret" | The three representative checks below. A later yes does not undo this |
| says anything else | "I'm the other insured", "I'm on the policy too", "Yes, I'm his wife and the policyholder", "why do you ask?" | Asked again for a plain yes or no, with the reason. After two more unclear answers a person takes over |

A yes is defined by what it may contain: yes words, an exact statement, and talk about the claim with no other person in it. Wording the rules have never seen is asked about and cannot open the gate. A model's reading can send a caller to the representative checks. It can never open the gate, and a model's "policyholder" counts only when the caller's own words are the exact statement.

"I'm her mother" and "I'm calling for my son" are asked about and never assumed either way. A policyholder says them about a child on their own plan, and a parent says them when calling for a grown child. A name given before the details ("David here.") and a statement such as "I work for Margaret" count as soon as the record is known. After verification, talk about family on the caller's own plan changes nothing. An outright contradiction ("the policy isn't mine", "I'm calling for Margaret", "I'm David") hands the call to a person.

### Representative callers and consent

Someone calling for a policyholder has to pass three checks: the policyholder's details match, the caller is listed as an authorized representative, and the policyholder approves a consent request. The consent status comes from `consent_scenarios.json` (choose `default` or `timeout` in the UI before starting a conversation). Until consent is approved the session stays in VERIFY_ID. A request that expires is not sent again: the caller is offered a person, and then handed to one.

A representative gives their own full name. A middle name or initial is fine ("David J. Chen" is the listed David Chen), and a typing mistake can be corrected. A different name after a refusal goes to a person. So does a caller who gives no name after several tries, or who then says they are the policyholder. No step repeats without end.

### Output guard

The first protection is structural: the model is given no record data before verification, and only the tool results of the current path after it. The guard is a second check on what the model wrote. A reply is released only if:

- every claim number, policy number, email address, phone number, web address, dollar amount, percentage, date, time and length of time in it is grounded. Grounded means it appears in this turn's facts, was said by the caller, or was in the facts of an earlier turn after verification. An amount may also be the sum or difference of two grounded amounts
- it contains no value held only in the records (identity numbers, the names of other policyholders and of listed representatives, denial reasons, requested documents)
- before verification, it does not say the caller is verified, that it can see a record, or what the status of a claim is
- it does not say that something was sent or that a transfer happened unless the workflow did that in this turn
- it does not promise an outcome ("it will be approved"), offer an action the assistant cannot take ("I can expedite that") or report one as done ("I've reopened the claim")

Otherwise the deterministic reply is sent and the rejection is shown in the inspector. The guard works on patterns. It covers the usual ways of writing these things, not every possible paraphrase, which is why it is the second line and not the first.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Token used when the browser does not send one |
| `SOP_MODEL` | `claude-opus-5-5` | Default model |
| `SOP_EFFORT` | `low` | Reasoning effort for both calls |
| `SOP_TODAY` | `2026-03-05` | Demo clock. A date, or `real` for the system date |
| `SOP_INSPECTOR` | `on` | `off` removes harness internals from API responses |
| `PORT` | `8000` | Port inside the container |

SOP limits live in `apps/insurance_claims/sop/common.py`: three failed identity attempts, three names that match no policy before a name counts as a failed attempt, two rounds of persuasion, a human offered at three out-of-scope messages and a transfer at four. One caller message is at most 2000 characters.

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
  state.py                   session state and the memory slots
  data.py                    read-only access to the sample records
  verification.py            three-of-five identity matching
  normalize.py               names, dates, phone numbers, SSN redaction
  tools.py                   claim, consent, email and hand-off tools
  guidance.py                document follow-up guidance from the fixtures
  understanding.py           the Understanding structure and its JSON schema
  nlu_llm.py, nlg.py         prompts for the two model calls
  nlu_rules.py               offline understanding
  guard.py                   output guard
  email_summary.py           summary composer
  fixtures/                  sample data
server/app.py                FastAPI app
server/static/index.html     test UI
scripts/demo.py              scripted conversations in the terminal, offline
scripts/live_check.py        the same conversations against the real model, with a spending cap
tests/                       1390 tests, all offline
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

The suite is offline. It never reads `.env` and removes any token from the environment, so it cannot spend money.

The suite covers the demo case, the verification gate, the bonus scenarios, scope control, claim resolution, the workflow paths, the email step, representative consent, harness enforcement (tool gating, illegal transitions), the output guard against leaking or overclaiming model replies, and the shape of the request sent to the Anthropic API. `tests/test_review_findings.py`, `tests/test_second_review.py` and `tests/test_final_review.py` hold a regression test for each problem found by the three code reviews before submission, and `tests/test_cold_check.py` and `tests/test_third_check.py` do the same for three independent checks of the finished version. `ruff check .` passes with the configuration in `ruff.toml`.

## Checked against the live model

`scripts/live_check.py` runs scripted conversations against the real API with a spending cap, and checks the workflow outcome of every turn (phase, verification, claim chosen, email sent, nothing protected said before verification). Those checks hold whatever wording the model chooses.

```bash
.venv/bin/python scripts/live_check.py --ping                          # one tiny call to check the token
.venv/bin/python scripts/live_check.py --scenarios happy,bonus         # capped at $0.50 by default
.venv/bin/python scripts/live_check.py --scenarios all --model claude-haiku-5-5 --max-usd 0.10
.venv/bin/python scripts/live_check.py --offline --scenarios all       # the same outcome checks on the offline engine, free
```

Twelve scenarios are included: the seven demo scripts plus five written for the model (spoken-style language such as "bday is march fifteenth nineteen eighty five", prompt injection before verification, an anxious caller with a missing document, a policyholder who mentions a relative, and a spouse who has three correct details but no consent). The offline mode runs nine of them. The other three need a model.

Results on 2026-11-20, all twelve scenarios, on the version before the role question became a standard step:

| Model | Turns | Outcome checks | Replies written by the model | Median time per turn | Cost per turn |
|---|---|---|---|---|---|
| `claude-opus-5-5` (default) | 59 | all passed | 56 of 59 | 4.6 s | about $0.02 |
| `claude-haiku-5-5` | 59 | all passed | 55 of 59 | 2.1 s | under $0.001 |

These runs used the scripts as they were then, 59 turns in all. Since then the role question is asked whenever the caller has not said who they are, which adds one turn to five of the scripts (64 turns now). The current version passes the same outcome checks on the offline engine (`--offline`, nine scenarios, 45 turns), and tests with a scripted model cover the model path of the new step: a model's reading can never open the gate. The current version has not been run against the live API. The replies not written by the model are the out-of-scope declines, which use fixed wording by design (three or four per run). The output guard rejected no reply in these runs. `claude-sonnet-5-5` (12 turns) and `claude-fable-5-2` (1 turn) passed on an earlier version and were not run again.

The Docker image was built and run: the container passes its health check, serves all seven demo scripts, and answers through the model when the token is passed with `-e`.

## Limits

- A safety classifier can decline a request on some models. For those models the client opts into the API's server-side fallback (`fallbacks: "default"`), and drops the option on its own if the account does not have it. If a model call fails in any way the turn still completes: the offline rules read the message, or the built-in wording is used for the reply, and the UI says which.
- The offline engine is English only and rule-based. It answers what its rules recognise and asks again otherwise.
- Memory keeps one claim reference at a time. Two questions about two different claims asked before verification are both remembered, and answered for the claim named last.
- The limit on failed identity attempts is per conversation. A new conversation starts a new count, and limiting attempts across conversations would need caller identification that this demo does not have.
- A policyholder is told which of their details matched ("I've confirmed your full name"), which makes verification easier to complete and also tells a guesser which values were right. A representative is not told. Wrong values are limited by the attempt count.
- The gate cannot detect a lie. Someone who knows three details and answers the role question with a plain yes is treated as the policyholder. What the gate rules out is a third party getting through by describing themselves truthfully, or by wording the rules did not expect.
- The role question needs an answer in English: an outright statement or a plain yes. A yes in another language, or with words next to it that cannot be placed, is asked again, also when a model is used. A policyholder who answers with more than a yes ("Yes, I'm the policyholder. My husband is covered under my plan.") is therefore asked once more.
- A caller who answers the role question as somebody's relative ("yes, I'm her husband") goes through the representative checks for the rest of the call. A policyholder who meant a dependant on their own plan is handed to a person when they then give the policyholder's name as their own. The agent does not settle that contradiction itself.
- A phone number or an email address counts as an identity detail only when it is offered as the caller's own ("my phone is ...", a value on its own, an item in a list of details). Mentioned in any other sentence it is not read, so it neither matches nor counts as a failed attempt.
- After verification the call goes to a person when the caller says outright that the policy is somebody else's or that they are somebody else. The usual ways of saying so are recognised, and some will be missed. Talk about family members does not hand the call off, since a policyholder has dependants.
- Sessions live in memory. Email delivery is simulated: the message appears in the Outbox panel.
- There is no hosted instance. The image runs on any container host and honours `PORT`.
