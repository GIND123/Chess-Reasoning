# Chess-Reasoning

**Process-verified reasoning: using step-level verification as a training signal, with chess as the testbed.**

Research plan and engineering specification of record.

**Status:** pre-training. 150,000 base prompts built, 11,796 annotated (superseded — see §1), 0 models trained.

---

## 0. Overview

### 0.1 The one-paragraph version

Synthetic reasoning corpora are built by showing a teacher model the answer and asking it to
explain. We measured what that produces: **19.5% of the resulting explanations assert something
about the board that is factually false**, and the teacher's own traces show it re-deriving the
position in 96.3% of cases because it cannot read the board it is reasoning about. Chess is the rare
domain where every intermediate claim in a reasoning trace — "the knight on f6 is pinned", "Rxe5
hangs the rook" — is checkable by a function call, at microsecond cost, with no human and no reward
model. We build that verifier, use it as both a data filter and a reinforcement-learning reward, and
show that **process-level verification beats outcome-level verification** on accuracy, on
faithfulness, and at test time. The result is a small open model that leads on grounded chess
reasoning and a method that transfers to any domain with a step-level checker.

### 0.2 Target results

Explicit hypotheses, stated before the experiments are run.

| Metric | Current best published | Target | Source of the comparison |
|---|---|---|---|
| Puzzle accuracy, open model ≤8B | 48.1% (C1-4B) | **≥ 55%** | Tang et al. 2026 |
| Claim-level factual precision, **any** model | 74.3% (Gemini, tools on) | **≥ 90%** | Hebbar et al. 2026 |
| Atomic recall, any model | 60.9% | **≥ 75%** | Hebbar et al. 2026 |
| Board-state reconstruction | 0.0% (all models) | **≥ 60%** | Wang et al. 2507.00726 |
| Strategic (near-tie) positions | **not reported by anyone** | first published numbers | — |
| Test-time verified reranking | does not exist | **+4–8 pts over greedy** | this work |

The defensible claim is **state of the art on grounded chess reasoning**: best open model at its
scale on move quality, and best model *at any scale* on the grounding and faithfulness metrics — a
regime where frontier systems currently sit at 53.9–74.3% precision and 44–61% recall. That second
claim is the paper's headline and it does not require out-scaling anyone.

### 0.3 Design commitments

Three commitments shape every experimental decision, and each is a methodological strength worth
stating in the paper:

1. **Matched scale.** Every trained system — ours and all five reimplemented comparisons — uses the
   same 4B base, the same data budget, and the same decoding configuration. Differences are
   attributable to method, not to scale. Prior work in this area compares 8B fine-tunes against
   commercial APIs, which confounds everything.
2. **Single-GPU reproducibility.** No experiment requires more than one 80GB GPU, and the entire
   data pipeline runs on a 24GB consumer card. The verifier is CPU-only.
3. **Deterministic verification.** No LLM sits inside the reward loop. Claim extraction is parsing;
   claim checking is `python-chess` and a precomputed engine table. Reward is reproducible bit-for-bit.

---

## 1. Empirical audit

All numbers measured directly with `python-chess` 1.11.2. Sample sizes and seeds in §15.1.

### 1.1 The base position set is sound — keep it

| Property | Value |
|---|---|
| Rows | 150,000 |
| Unique FENs | 149,982 (0.01% duplicate) |
| Invalid FENs / illegal `Best_move` | **0 / 0** in 20,000 sampled |
| Side to move | 50.7% white, 49.3% black |
| Phase mix | 69.6% middlegame, 24.4% endgame, 6.0% opening |
| Game-level leakage | **none** — 149,897 distinct pawn skeletons, 0.1% shared |

The last row matters. Chess corpora are normally built from consecutive plies of the same games,
which makes a random position-level split leak. Ours does not, so a position-level split is
defensible — state this explicitly in the paper, since reviewers will assume otherwise.

### 1.2 Eight defects in the labelling pipeline

**F1 — The target is inside the input, in 100% of rows.** Every prompt ends with the literal best
move, and across 30,000 parsed prompts **Line 1 begins with the correct move 30,000 times out of
30,000**. Deleting one sentence is insufficient; the engine block leaks the answer too.

**F2 — The "Grandmaster annotation" is 200 template strings that leak the move type.** Exactly 200
unique strings across 150,000 rows; 20 cover 66.8%, 50 cover 81.1%, no singletons. They are mutual
paraphrases with no positional content (*"Advance the piece to a strategic square, optimizing its
control over the board"*, 3.48%). This matches MATE's own description — expert-authored *rules* with
"approximately 20 distinct linguistic expressions" per category. **It must not be described as a
Grandmaster annotation in a paper.** And it is not inert:

| Property of the correct move | MI with template | Feature entropy | Share explained |
|---|---:|---:|---:|
| Is a capture | 0.732 bits | 0.74 | **98.4%** |
| Piece type moved | 0.694 bits | 2.45 | 28.3% |
| Gives check | 0.185 bits | 0.97 | 19.1% |

The hint tells you almost deterministically whether the answer is a capture. Drop it from the
prompt; keep it as ablation **A6**.

**F3 — The position distribution is not chess.** Line-1 eval favours the mover in **99.3%** of
positions; median eval **+464 cp**; 25th percentile **+371 cp**; **29.6%** (44,395) have a forced
mate as the best line; median best-vs-second gap **576 cp**; only **0.2%** of gaps fall below 200 cp.
This is upstream by design — MATE selects move pairs whose score difference exceeds a threshold. The
corpus contains almost no *chess decisions*, only tactical shots in won positions. See §4.4.

**F4 — 19.5% of generated answers state a false board fact.**

| Check | Checked | Wrong | Rate |
|---|---:|---:|---:|
| Piece-on-square claims | 7,596 | 633 | **8.3%** |
| Answers with ≥1 false claim | 3,000 | 585 | **19.5%** |
| UCI moves referenced | 1,760 | 25 | 1.4% |
| Answers referencing an illegal move | 3,000 | 23 | 0.8% |

A **lower bound** — it catches only explicit "piece on square" phrasing and bare UCI. Pins, forks,
attack relations and material claims are unchecked and are where most remaining errors live.

**F5 — The generator cannot read the board.** Across all 10,296 hidden `<think>` traces: **100%**
contain a self-correction marker, mean **18.5 per trace** (median 19, max 54); **96.3%** contain an
explicit board re-derivation. Worked example, row 2, position
`8/2Q1bRpk/p6p/1p6/8/2P5/PPbr2RP/7K b - - 0 25`, black king on **h7**:

> "Rank 7: `2Q1bRpk` → Q on c7, b on e7, R on f7, p on h7, k on **h8**."

Wrong twice in one line. The model then reasons about a mating net around a king that is not there,
and still emits the correct move — because the correct move was printed in the prompt. That is the
entire failure in one row. It matches the published **0.0% board-state comprehension** result across
all tested models. Treat it as the central scientific finding: *board grounding, not search depth,
is the binding constraint.*

**F6 — 26.0% of answers cite the template hint**, in sentences unconditioned on the board that
become nonsense once the hint is removed.

**F7 — Stylistic collapse.** Distinct-6 = 0.701; 12.0% of answers share the 6-gram *"is the
strongest continuation as it"*; about a third open with one of four fixed frames. This flattens GRPO
rollout diversity and suppresses the reward variance the algorithm depends on.

**F8 — 96% of generated tokens discarded; one column silently empty.** Mean hidden trace 14,161
chars vs 771 chars kept — a ratio of **18.4×**. `extracted_reasoning` is empty in **10,278 of
10,296 rows** (99.8%). Root-cause before regenerating.

### 1.3 Two tooling consequences

- **GGUF cannot be trained.** `Qwen3.5-27B-Q5_K_M` under llama.cpp is an inference format. Training
  requires BF16 HF weights plus a separate vLLM engine for rollouts.
- **The model-selection benchmark measured the wrong thing.** The LLM judge scored prose similarity
  to a gold explanation, not correctness. Re-run selection with the verifier (§5) as the metric —
  claim precision and move accuracy. It converts a soft benchmark into a hard one at negligible cost.

---

## 2. Thesis and contributions

> **Thesis.** Reasoning traces produced by answer-conditioned teachers are systematically unfaithful
> to the state they claim to reason over. Where a step-level verifier exists, this is measurable and
> correctable, and **process-level verification dominates outcome-level verification** as a training
> signal. Chess is the domain where this can be demonstrated rather than argued, because every
> intermediate claim reduces to a function call.

Why chess is the right substrate for a reasoning-methods paper:

- **Steps are checkable.** In mathematics and code only the final answer is verifiable, which is why
  essentially all RLVR is outcome-level. Chess admits cheap, deterministic process supervision.
- **The oracle is graded, not binary.** Stockfish scores *every* legal move continuously, so
  positions with several defensible answers remain trainable — the regime where binary correctness
  is undefined.
- **The failure is large and pre-measured.** 0.0% board-state accuracy across published models; 8.3%
  false claims in our own corpus. There is a wide, visible gap.
- **Difficulty is ground-truth.** Puzzle rating and eval gap give an objective stratification axis.

### Contributions

1. **A measurement.** The first quantification of unfaithfulness in an answer-conditioned synthetic
   reasoning corpus, with a matched control (§7, M1).
2. **An instrument.** A deterministic, CPU-only, twelve-class chess claim verifier, released as a
   package. Reusable independent of the model.
3. **A method.** Process-verified GRPO — claim precision and coverage as reward terms alongside
   graded move quality (§6.2).
4. **A test-time result.** Verified reranking: select among sampled traces by claim-verification
   score alone, with no engine and no ground truth (§6.3).
5. **A controlled comparison.** Five prior methodologies reimplemented at matched scale on a common
   benchmark (§7) — the first like-for-like comparison in this area.

---

## 3. Related work and positioning

| Work | Method | Headline | What it leaves open |
|---|---|---|---|
| **Wang et al., NAACL 2025** (MATE — our data source) | 1M annotated positions; SFT Llama-3-8B; binary two-move choice | 95.2% *with explanations in prompt* | Forced choice with the justification supplied. 63.5% without explanations — barely above the 50% floor. No legality, no playing strength, no open-ended generation. |
| **Wang et al., arXiv 2507.00726** | GRPO on 19.2k Lichess puzzles; dense reward from a 270M action-value network | 25–30% puzzle accuracy | Plateaus far below the 66.5% human-expert bar. Diagnosis — models "lack fundamental chess understanding" — asserted, not fixed. **0.0% board-state accuracy.** |
| **Tang et al., arXiv 2603.20510** (Toronto) | Master Distillation: Stockfish d24 PVs + Gemini traces via Feigned Discovery Prompting → SFT Qwen3-4B → DAPO, binary reward | 48.1% vs 40.9% SFT-only | Stated limitation, verbatim: a real system "must generalize to **strategic positions where multiple reasonable continuations exist**." Traces "may not faithfully capture underlying logic" — and faithfulness is never measured. |
| **Hebbar et al., arXiv 2608.04240** (ACT-Eval) | Atomic-claim decomposition, 15+ board tools, Stockfish d18 | 53.9–74.3% factual precision | **Builds the verifier and never trains on it.** Purely evaluative. Atomic recall stalls at 44–61% even with tools at inference. |
| **ChessQA** (arXiv 2510.23948, CSSLab) | Five-category chess understanding benchmark: structural, motifs, short tactics, position judgment, semantic | — | Adopted as an evaluation suite (§9). Piece/square recognition and legality are the published weak points. |
| **Brittleness testing** (arXiv 2605.17565) | Generalization-vs-memorization probes for chess-trained LMs | — | Directly motivates our contamination controls (§9); adopt its perturbation protocol rather than inventing one. |
| **LLM CHESS** (arXiv 2512.01992) | Agentic full-game benchmark; instruction-following under extended play | Clear separation between reasoning and non-reasoning models | Adopted for Tier 4. Many SOTA models cannot complete a game against a weak opponent. |
| **Myopic planning** (arXiv 2605.06840) | Extracts search trees from reasoning traces | Traces reveal shallow, myopic lookahead | A ready-made analysis for our traces: does process-verified training deepen the extracted search tree? |

**The gap, precisely.** One group built a claim verifier and used it only as a metric. Another
trained with RL but rewarded only the final move, on puzzles, and explicitly deferred strategic
positions. **Nobody has closed the loop.** Two specific inheritances to acknowledge honestly:
Tang et al. independently identified the rationalization problem and built Feigned Discovery
Prompting against it — we cite them for the problem and contribute *verifying rather than trusting*
the result, since F4 shows discovery-form traces are frequently still false. And ACT-Eval's tool
suite is the direct ancestor of our verifier; our contribution is turning it into a gradient.

---

## 4. Corpus construction

### 4.1 Datasets

| Set | Size | Source | Purpose |
|---|---:|---|---|
| **A — Tactical** | 150,000 | existing `GRPO_GM_dataset.csv` | SFT bulk; verifier bootstrapping |
| **B — Strategic** | 40,000 | new, Lichess DB, eval-banded | RL positions; the novel regime |
| **C — Leaked control** | 11,796 | existing `10k_chunk_1.csv` + regeneration | Comparison M1 |
| **D — Held-out** | 6,000 | banded, contamination-controlled | Never touched until §9 |

### 4.2 Trace format

The current corpus explains only *why the given move is good* and says nothing about why the
alternatives lose. That is the single most important omission: a model trained on it has no basis
for discriminating between moves, which is the entire task at inference. Replace it with a
**contrastive candidate-elimination trace**:

```
<read>
Two or three sentences of concrete position facts: material, king squares,
immediate threats, key squares. Verifiable statements only.
</read>
<candidates>
1. <uci> — <concrete continuation> — <verdict>
2. <uci> — <concrete continuation> — <verdict>
3. <uci> — <concrete continuation> — <verdict>
</candidates>
<choice>
One sentence: the chosen move and the single decisive reason.
</choice>
<move>e2e4</move>
```

Rationale: `<read>` enforces and exposes board grounding, targeting F4/F5 directly. `<candidates>`
carries the contrastive signal, with at least one candidate required to be a plausible mistake with
a concrete refutation. `<move>` makes reward and accuracy extraction a regex, never an LLM call. The
structure keeps claim extraction ~90% deterministic parsing, which matters because ACT-Eval flags
LLM decomposition as an error source it could not eliminate. Target 120–220 words (~200–320 tokens):
Tang et al. made token efficiency a headline (178 tokens vs 12,193 for GPT-5), and our current
traces are 14,161 chars of confused backtracking (F8).

### 4.3 The two prompts

**Teacher** (generation only — sees engine data):

```python
SYS_TEACHER = """You write chess analysis for a training corpus. You are given a position and
engine analysis of the candidate moves. The engine data is PRIVATE CONTEXT: it tells you what is
true, but it must never appear in your output, and your output must read as analysis produced by a
strong player seeing this position for the first time.

Hard rules:
1. Never mention the engine, evaluations, centipawns, scores, "the best move is", or the fact that
   you were given an answer.
2. Every factual statement about the board must be true of the position exactly as given. Name
   pieces by square. If you are not certain a piece is on a square, do not mention it.
3. Consider exactly three candidate moves. At least one must be a plausible mistake that a strong
   human would seriously consider, and which the engine data shows is inferior.
4. For each candidate give the concrete continuation and the concrete reason it succeeds or fails:
   a specific recapture, a specific square, a specific material or mating consequence. Never
   "this is passive" or "improves coordination" without a concrete follow-up.
5. Output the structure below and nothing else."""

USER_TEACHER = f"""{ascii_board}

FEN: {fen}
{side} to move.

PRIVATE engine analysis (never reference):
{move_table}          # every legal move with its evaluation, best first

Write the analysis."""
```

**Student** (stored in the SFT pair; used at RL and inference):

```python
USER_STUDENT = f"""{ascii_board}

FEN: {fen}
{side} to move.

Analyse the position and choose the best move."""
```

The teacher sees the engine table; **the student never does**. The stored example is
`(USER_STUDENT → trace + <move>)`. This closes F1 at zero cost.

### 4.4 Position banding

Set A cannot support the central claim because it has no close decisions (F3: 0.2% under 200 cp).
Set B is sourced fresh from the Lichess standard database and stratified by decision difficulty:

| Band | Best-vs-second gap | Target share | Present in Set A |
|---|---|---:|---:|
| Near-tie | < 30 cp | 40% | ~0% |
| Moderate | 30–100 cp | 30% | ~0% |
| Decisive | 100–300 cp | 20% | 0.2% |
| Tactical | > 300 cp, mates | 10% | 99.8% |

The near-tie band is where GRPO becomes interesting: several rollouts can be legitimately good, so
the algorithm must separate them on *reasoning quality* rather than memorise a label.

**Precondition to verify before committing:** within-group reward standard deviation must exceed
~0.05 in every band, or group-relative advantages collapse to zero and the gradient vanishes.

### 4.5 Engine tables

For every position, one Stockfish pass with MultiPV over **all** legal moves at a fixed **node**
limit (not time — node limits are reproducible, time limits are not), storing
`{uci → (cp, win_prob)}`. Reward and consequence-claim lookups are then dict accesses and no engine
runs inside the training loop. ~33 core-hours; runs locally overnight on any multi-core machine.

Three fixes to `Annotation_generator/annotator.py` while doing this:

1. `_get_top_lines` opens and closes a Stockfish process **per position** — use a persistent pool
   per worker; process spawn currently dominates runtime.
2. `Limit(time=2.0, depth=30)` → node limit for reproducibility.
3. The three top lines are **already embedded in `GRPO_GM_dataset.csv`** and can be parsed out free
   for teacher context. Only the full-table pass is new work.

### 4.6 Acceptance, repair, retry

| Gate | Rule | On failure |
|---|---|---|
| Format | `<move>` parses to a legal UCI | reject |
| Move quality | win-prob within `TOL` of best | reject (Set A) / graded label (Set B) |
| Occupancy | **zero** false piece claims | reject — this is the defect being removed |
| Legality | zero illegal move references | reject |
| Claim precision | ≥ 0.9 over verifiable claims | flag, keep |
| Coverage | ≥ 2 verifiable claims in `<read>` | reject — guards against empty grounding |

Adaptive sampling: `k=1` across Set A, measure acceptance, then re-run only rejects at `k=4` with
higher temperature. The acceptance curve as a function of position difficulty is itself a reportable
figure.

**Do not repair traces with a second LLM pass.** It reintroduces exactly the unverified content the
filter exists to remove. Reject and resample.

---

## 5. The verifier

The core instrument. Deterministic, CPU-only, no model in the loop.

| # | Claim type | Check | Cost |
|---|---|---|---|
| 1 | Piece occupies square | `board.piece_at()` | µs |
| 2 | Move is legal | `move in board.legal_moves` | µs |
| 3 | Move is capture / gives check | `is_capture()`, `gives_check()` | µs |
| 4 | Square attacked / defended by side | `board.attackers()` | µs |
| 5 | Piece is pinned | `board.is_pinned()` | µs |
| 6 | Material balance after a line | push line, count | µs |
| 7 | Piece mobility ("has no squares") | legal moves from square | µs |
| 8 | Pawn structure (doubled, isolated, passed) | pawn bitboards | µs |
| 9 | Line is forced | cached engine table | lookup |
| 10 | Consequence ("after X, White is winning") | cached engine eval | lookup |
| 11 | Move hangs material | cached engine delta | lookup |
| 12 | Vague strategic claim | **unverifiable — counted, not scored** | — |

Types 1–8 need only `python-chess` — no engine, no ground truth. That is what makes §6.3 possible.

```python
import re, chess

PIECE = {'king':chess.KING,'queen':chess.QUEEN,'rook':chess.ROOK,
         'bishop':chess.BISHOP,'knight':chess.KNIGHT,'pawn':chess.PAWN}

OCCUPANCY = re.compile(r'\b(?:(white|black)\s+)?(king|queen|rook|bishop|knight|pawn)s?\s+'
                       r'(?:on|at)\s+([a-h][1-8])\b', re.I)
UCI       = re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b')
PINNED    = re.compile(r'\b(king|queen|rook|bishop|knight|pawn)\s+on\s+([a-h][1-8])'
                       r'\s+is\s+pinned\b', re.I)

def verify(fen, text, engine_table=None):
    """Returns (n_claims, n_true, violations). Deterministic; no model calls."""
    b = chess.Board(fen)
    claims = true = 0
    bad = []

    for m in OCCUPANCY.finditer(text):
        colour, name, sq = m.group(1), m.group(2).lower(), m.group(3).lower()
        pc = b.piece_at(chess.parse_square(sq)); claims += 1
        ok = (pc is not None and pc.piece_type == PIECE[name]
              and (colour is None or pc.color == (colour.lower() == 'white')))
        true += ok
        if not ok: bad.append(('occupancy', m.group(0)))

    for m in UCI.finditer(text):
        u = m.group(1); claims += 1
        try:
            mv = chess.Move.from_uci(u)
            ok = mv in b.legal_moves or (len(u) == 4 and
                 chess.Move.from_uci(u + 'q') in b.legal_moves)
        except ValueError:
            ok = False
        true += ok
        if not ok: bad.append(('illegal_move', u))

    for m in PINNED.finditer(text):
        sq = chess.parse_square(m.group(2).lower())
        pc = b.piece_at(sq); claims += 1
        ok = pc is not None and b.is_pinned(pc.color, sq)
        true += ok
        if not ok: bad.append(('pin', m.group(0)))

    # ... attack/defence, material, mobility, structure, engine-backed types 9-11 ...
    return claims, true, bad
```

### 5.1 Conservative move handling — a measured design decision

The first implementation verified every move mentioned in a trace against the root position. Run
over 3,000 legacy answers it flagged 255 illegal move references — and inspection showed **221 of
them were extractor false positives**, not model errors. Almost all were moves *inside a variation*:

> "…which subsequently allows Black to capture the queen with **Rxc1**…"
> "If White defends with **Re1**, Black responds with **Qxe1**…"

That is correct analysis of a later position. Scoring it against the root would have trained the
model to stop writing variations. A second class came from splitting long algebraic `Rd8-d5` into a
spurious SAN move `Rd8`.

The verifier therefore applies the rules below, each pinned by a test:

- Moves are root-scored **only** inside `<candidates>` heads and the `<move>` tag; every other move
  reference is extracted and marked `UNSCORED`.
- Continuations are verified by **replay** (`verify_line`) — the correct check for a variation.
- Long algebraic is matched before SAN so its head is never re-read as a move.
- Bare squares are never treated as moves.
- `covered` was dropped from the defended-synonym set as ambiguous; attack-claim precision on the
  legacy corpus rose from 53.3% to 83.3%.

This is the concrete reason the plan builds the verifier before the model.

### 5.2 Two non-negotiable properties

- **Extractor precision must be hand-validated** on ~200 traces before it feeds a gradient. Target
  ≥90% agreement. `scripts/00_validate_extractor.py --sample` emits the labelling sheet and
  `--score` reports the gate. A noisy extractor makes the reward noise.
- **The `unverifiable` bucket must exist**, counted but not scored, so vague prose is neither
  rewarded nor punished as though it were false.

The existing 10,296 answers are a free test fixture — the expected occupancy result is already known
(8.3% false, F4).

---

### 5.3 Codebase

Implemented and tested: `pytest -q` → 48 passing. The verifier doubles as its own regression test —
`make audit` must report an occupancy error rate of ~8.3%, the value measured in §1.2.

```
src/chessr/
  boards.py      FEN/ASCII utilities; the win-probability scale used by every reward
  prompts.py     teacher + student prompts; structured-trace parser
  claims.py      THE VERIFIER - 12 claim types, deterministic, CPU-only
  reward.py      reward components + TRL-compatible reward functions
  engine.py      Stockfish pools, full-legal-move tables, TableStore
  datasets.py    Sets A-D, banding, splits, datacard statistics
  generate.py    vLLM offline batching (no server, no client)
  filtering.py   acceptance gates
  sft.py         LoRA SFT on verified traces
  grpo.py        process-verified GRPO; reward composition driven by config
  rerank.py      verified reranking - engine-free, usable at test time
  evaluate.py    Tier 1-3 metrics per band; perturbation + corruption probes; bootstrap CIs
configs/         one YAML per arm: grpo_m6 (ours), m3, m4, m2, a3, sft, data
scripts/         00 validate extractor ... 06 evaluate
modal_app.py     Modal entrypoints: smoke, generate_all, grpo, evaluate_model
tests/           48 tests against hand-checked positions
```

**Quickstart**

```bash
pip install -e ".[train,dev]"
pytest -q                      # 48 tests
make audit                     # verifier regression against the legacy corpus

make tables WORKERS=8          # Stockfish tables over all legal moves, node-limited
make generate                  # pass 1, n=1
make filter                    # acceptance gates + rejection breakdown (week-5 gate)
make retry                     # pass 2, n=4, rejects only
make sft
python scripts/05_grpo.py --config configs/grpo_m6.yaml
make eval

modal run modal_app.py::smoke  # week-1 gate: 1000 prompts, one batch, no HTTP
```

**Adding an arm is a config file, not a code change.** `terms` and `weights` in the GRPO config
select which reward components are active, so M2/M3/M4/A1/A3 differ from M6 only by YAML. That is
what keeps the comparison honest.

---

## 6. Training

### 6.1 Stage 1 — supervised fine-tuning

- **Policy:** `Qwen3-4B-Instruct`, BF16, LoRA (r=32, α=64, all linear). Matches Tang et al.'s base
  exactly, so the comparison in §7 is like-for-like.
- **Teacher:** `Qwen3-14B-AWQ` with the engine table in context. The teacher's task is *verbalising
  a known answer*, not solving the position, so teacher scale matters far less than the verifier's
  strictness. Re-validate teacher choice with the verifier, not the LLM judge (§1.3).
- **Data:** ~100k accepted traces from Sets A+B. Tang et al. saw no saturation at 39k; treat that as
  a floor.
- Train the M1 leaked arm in parallel — it costs a fraction and is a headline comparison.

### 6.2 Stage 2 — process-verified GRPO

```
R = w₁·R_move + w₂·R_precision + w₃·R_coverage + w₄·R_format
    − hard penalties (illegal move, false occupancy claim)
```

**R_move — quality in win probability, not centipawns.** Centipawns are badly non-linear: 50 cp is
decisive at equality and irrelevant at +900. With 99.3% of Set A above +300, a cp-based reward would
be nearly flat there and noisy elsewhere.

```python
def wp(cp):
    return 1 / (1 + 10 ** (-cp / 400))          # standard logistic win probability

R_move = 1 - clip((wp(cp_best) - wp(cp_played)) / TOL, 0, 1)     # TOL ≈ 0.10
```

This makes a 20 cp inaccuracy at equality cost real reward and the same inaccuracy in a won position
cost almost none — correct chess and the right gradient. It is also what makes the near-tie band
trainable: several moves score 0.9+, so within-group spread must come from the grounding terms.

**R_precision** — verified claims / asserted claims. Drives the measured 8.3% false-claim rate toward
zero, and is directly comparable to ACT-Eval's published factual precision.

**R_coverage — not optional.** A policy maximising precision alone learns to say almost nothing:
assert one true fact, stop. Pair precision with recall against the position's salient content — the
motif present, the key defended square, the refutation of the strongest alternative — with the
reference claim set derived from the engine table so no human labelling is needed.

> **Without R_coverage the reward has a degenerate optimum and the run will find it.** This is the
> most likely quiet failure: the training curve looks healthy while traces get shorter and emptier.
> Inspect top-reward rollouts every 100 steps.

**Hard constraints.** Illegal moves and false occupancy claims are penalties, not weighted terms.
Report illegal-move rate **against a constrained-decoding control** (A8) — legality is free via
logit masking, and a reviewer will ask.

**Why RL at all, when SFT data is already verifier-filtered?** Two reasons, and the second is the
real one. RL pushes past the teacher — Tang et al.'s 4B student beat its Gemini teacher, 48.1% vs
40.8%. And **on Set B strategic positions there is no single gold trace to imitate**, so SFT is
ill-posed and graded RL is the only available method.

**Configuration:** 400 steps, 24 prompts/step, group size G=8, max 400 new tokens, no KL penalty
(DAPO-style; removes the reference model), LoRA on the policy, vLLM colocated with sleep/wake weight
sync. ~30.7M rollout tokens per run.

### 6.3 Stage 3 — verified reranking at test time

Claim types 1–8 need only `python-chess` — **no engine, no answer, no ground truth**. So at
inference: sample *n* traces, score each by claim verification, return the move from the
highest-scoring trace.

If faithfulness predicts correctness, this is accuracy obtained from a signal available at test
time. Report the faithfulness–accuracy correlation directly (per-position verification score vs move
correctness). It is the cleanest evidence that grounding is *causal* rather than cosmetic, and it is
a new inference-time method rather than a re-run of self-consistency.

A second framing worth making explicit: ACT-Eval shows tools help *at inference*. Training against
tool-verified rewards **distils the verifier into the policy**, so the tools are needed at training
time only. A8/A5 test whether that holds.

---

## 7. Comparison methodologies

Five prior methodologies reimplemented at matched scale (4B base, same data budget, same decoding),
plus ours. This is the first like-for-like comparison in this area — existing papers compare
fine-tuned small models against commercial APIs, which confounds method with scale.

| ID | Methodology | Origin | Implementation |
|---|---|---|---|
| **M1** | Answer-conditioned annotation SFT | Wang et al., NAACL 2025 (MATE) | SFT on Set C — the existing leaked corpus, unchanged |
| **M2** | Master Distillation | Tang et al., 2026 | Feigned-discovery SFT (no verification filter) + binary-correctness RLVR, DAPO-style |
| **M3** | Dense action-value RLVR | Wang et al., 2507.00726 | GRPO with graded move-quality reward only (our R_move, no grounding terms) |
| **M4** | Sparse outcome RLVR | DeepSeek-R1 style | GRPO with binary correct/incorrect reward |
| **M5** | Tool-augmented inference | Hebbar et al., 2026 (ACT-Eval) | Base + SFT model with board tools available at inference, no RL |
| **M6** | **Process-verified GRPO (ours)** | this work | Composite reward, §6.2 |

Additional non-trained reference points: base model zero-shot; base model with CoT prompting; a
no-reasoning direct-move policy; and frontier API models on a 1,000-position subset (for calibration
against published numbers, not as a matched comparison).

**What each comparison isolates**

- M1 vs M6 — the full effect of the pipeline redesign.
- M2 vs M6 — verification vs trusting a discovery-form teacher. *This is the closest prior system
  and the most important row in the table.*
- M3 vs M6 — grounding reward terms over and above a graded move reward.
- M4 vs M3 — graded vs binary outcome reward, replicating the published finding that dense beats
  sparse.
- M5 vs M6 — tools at inference vs the verifier distilled into the policy.

---

## 8. Ablation program

| ID | Ablation | Arms | Isolates |
|---|---|---|---|
| **A1** | Reward composition | R_move / +R_prec / +R_cov / full | Contribution of each reward term |
| **A2** | SFT data provenance | leaked / discovery-unverified / discovery-verified / +contrastive | Which data change matters, and how much |
| **A3** | Verifier strictness | precision-only reward | Demonstrates the degenerate optimum §6.2 warns about |
| **A4** | Policy scale | 1.7B / 4B | Whether the effect is scale-dependent |
| **A5** | Test-time reranking | n ∈ {1, 4, 8, 16} | Verified reranking gain curve |
| **A6** | Template hint | present / absent | Quantifies the F2 shortcut (98.4% MI leak) |
| **A7** | Position banding | tactical-only / balanced | Whether strategic positions are necessary or merely novel |
| **A8** | Legality masking | on / off | Separates learned legality from decoded legality |
| **A9** | Trace length budget | 200 / 400 / 800 tokens | Token efficiency vs accuracy frontier |

A2, A4, A5, A6, A9 require no additional RL runs — they are SFT-level or inference-only, and run on
local hardware.

---

## 9. Evaluation protocol

Existing work evaluates narrowly and is criticised for it — MATE reports two-way choice accuracy,
the RL papers report puzzle pass@1, and **none of them plays chess**.

**Tier 1 — move quality**
- Top-1 engine agreement, **reported per band**, never pooled. Pooling is how the F3 skew inflates a number.
- Mean win-probability loss — the ACPL analogue, and the metric a chess audience actually reads.
- Illegal-move rate, with and without constrained decoding.

**Tier 2 — grounding**
- Claim-level factual precision and atomic recall under the ACT-Eval protocol, so numbers sit directly beside theirs.
- ChessQA structural and legality categories — the published weak spot, and where the largest gain should appear.
- Board-state reconstruction after an *n*-ply line: the task where every model currently scores 0.0%.

**Tier 3 — faithfulness**
- **Counterfactual perturbation:** relocate one piece so a stated justification becomes false. Does the chosen move change? A rationalizing model will not move.
- **Trace corruption:** corrupt a verified claim mid-trace, measure the answer shift.
- All of it reported against M1. That contrast is the paper's cleanest figure.

**Tier 4 — playing strength**
- Elo vs Stockfish at pinned skill levels and node limits, ≥400 games for a usable confidence interval.
- Round-robin against the open baselines.
- Blunder rate per 100 moves — chess readers trust this more than puzzle accuracy.

**Tier 5 — transfer**
- Does process-verified chess training move general state-tracking and multi-step reasoning benchmarks?
- Check for catastrophic forgetting on general instruction-following.
- Honest either way. A null result is a contribution given how loudly domain-RL transfer is currently claimed.

**Contamination control.** MATE positions come from published Lichess games and puzzles predating
every model's cutoff. Include a split from games played after the base model's cutoff plus a
synthetic-position split, and report the gap. If accuracy collapses on post-cutoff positions that is
itself a finding — better discovered by us than by a reviewer.

**Error analysis (journal requirement).** Hand-code 200 failures from the best model into the
ACT-Eval taxonomy — board comprehension, legality, quality evaluation, tactical reasoning, other —
and report the distribution before and after process-verified training. This is what turns a results
table into an explanation.

---

## 10. Statistical protocol

Under-specified statistics are a common journal rejection reason in this area; prior work in §3
reports bare accuracies on 900–1,000 samples with no intervals.

- **Eval size:** 6,000 held-out positions (Set D), ~1,500 per band.
- **Uncertainty:** BCa bootstrap 95% CIs over positions, 10,000 resamples, for every reported metric.
- **Paired comparisons:** all systems evaluated on identical positions; use paired bootstrap and
  McNemar's test for accuracy differences. Report effect sizes, not only p-values.
- **Multiple comparisons:** Holm–Bonferroni across the primary comparison family (M1–M6 on Tier 1
  and Tier 2 headline metrics). Ablations are exploratory and labelled as such.
- **Seeds:** 2 training seeds on M6 and on the closest comparison (M2); 1 seed elsewhere, with
  seed-variance estimated from the M6 pair and reported as a caveat.
- **Decoding:** identical sampling parameters across all systems; temperature 0 for headline numbers,
  with the n>1 reranking results reported separately (A5).

---

## 11. Compute and infrastructure

### 11.1 Placement

Work is placed on the hardware that fits it. The verifier and engine tables are CPU-only; bulk
generation and SFT fit a 24GB card; only RL and the multi-model evaluation sweep need an 80GB device.

| Stage | Hardware | Notes |
|---|---|---|
| Engine tables (all legal moves) | Local CPU | ~33 core-hours, overnight |
| Verifier development + validation | Local CPU | No GPU at any point |
| Set A + B trace generation | Local RTX 3090, vLLM | ~6 h/pass tuned (§11.3) |
| SFT (M1, M2, M6 arms) | Local RTX 3090, LoRA | Overnight per arm |
| A2, A4, A5, A6, A9 ablations | Local RTX 3090 | No RL required |
| **GRPO runs (M2, M3, M4, M6, A1, A3)** | **Modal, H100** | ~2.5 h/run |
| **Baseline + held-out eval sweep** | **Modal, L40S** | Multi-model, Volume-cached weights |
| **Elo game play** | **Modal, L40S** | 400+ games, parallel |

### 11.2 The generation architecture

Every failed approach in the deployment notes — Ollama single-stream, Ollama on Modal, Colab CUDA
builds, vLLM server plus separate client — shares one assumption: that generation needs a
long-lived HTTP server another process connects to. **It does not.** vLLM's offline `LLM` class
batches in-process. No port, no service discovery, no client, and the entire failure class
disappears rather than being debugged.

```python
import modal
app = modal.App("chess-gen")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("shards",   create_if_missing=True)

image = (modal.Image.debian_slim()
         .pip_install("vllm==0.11.0", "python-chess")
         .env({"HF_HOME": "/cache", "VLLM_USE_V1": "1"}))

@app.function(image=image, gpu="L40S", volumes={"/cache": vol, "/out": out},
              timeout=6*3600, retries=2)          # no region pinned: avoids the region multiplier
def run_shard(shard_id: int):
    import os
    if os.path.exists(f"/out/{shard_id}.jsonl"):
        return                                     # idempotent: a crash costs one shard
    from vllm import LLM, SamplingParams
    llm = LLM("Qwen/Qwen3-14B-AWQ",
              quantization="awq",
              max_model_len=1280,                  # see §11.3 - the dominant lever
              kv_cache_dtype="fp8",
              enable_prefix_caching=True,
              gpu_memory_utilization=0.92)
    outs = llm.generate(load_shard(shard_id),
                        SamplingParams(n=1, temperature=0.7, max_tokens=400))
    write_atomic(f"/out/{shard_id}.jsonl", outs); out.commit()

@app.local_entrypoint()
def main():
    list(run_shard.map(range(64)))                 # fan-out, zero inter-container communication
```

The identical script runs locally by dropping the Modal decorators.

### 11.3 vLLM tuning — where the throughput actually comes from

The dominant lever is **`max_model_len`**, and it is free. Our prompts are ~450 tokens and traces
are capped at 400, so a 1,280-token window is ample. vLLM allocates KV blocks against
`max_model_len`, so leaving it at the model default starves concurrency:

**Qwen3-14B-AWQ on a 24GB card** (160 KB KV/token, 13.1 GB free for KV):

| `max_model_len` | KV dtype | Concurrent sequences |
|---:|---|---:|
| 8192 (default) | bf16 | 9 |
| 8192 | fp8 | 19 |
| 2048 | bf16 | 38 |
| 1280 | bf16 | 62 |
| **1280** | **fp8** | **124** |

**A 14× concurrency gain from two configuration flags.** In wall-clock terms, 150k traces at 320
output tokens goes from ~14.8 h untuned to ~6.1 h tuned on the same hardware.

The same arithmetic on an H100 80GB with the 4B policy gives 695 concurrent sequences at
`max_model_len=1280` + fp8 KV, against 54 at defaults — which is what makes GRPO rollouts cheap.

Other settings that earn their place:

- `enable_prefix_caching=True` — the system prompt is shared across every request.
- `n=k` in `SamplingParams` rather than k separate calls — prefix compute is shared across the group,
  which matters for both rejection sampling and GRPO rollouts.
- **Guided decoding** for the `<move>` tag, constrained to the legal-move list via vLLM structured
  outputs. Free legality, and the control arm for A8.
- For GRPO: TRL `GRPOTrainer` with `use_vllm=True, vllm_mode="colocate"` and sleep/wake cycling, so
  trainer and engine share one device instead of needing two.
- Do **not** pin a Modal region — region selection applies a 1.5–1.75× rate multiplier.

### 11.4 Resource plan

| Item | Device | Hours | Est. |
|---|---|---:|---:|
| GRPO — M6 ours, 2 seeds | H100 | 5.0 | $21 |
| GRPO — M3 dense move-only | H100 | 2.5 | $11 |
| GRPO — M4 sparse binary | H100 | 2.5 | $11 |
| GRPO — M2 Master-Distillation repro | H100 | 2.5 | $11 |
| GRPO — A1/A3 reward ablation | H100 | 2.5 | $11 |
| Baseline + held-out eval sweep | L40S | 6.0 | $12 |
| Elo game play, 400+ games | L40S | 4.0 | $8 |
| Reserve for reruns | — | — | $20 |
| **Modal total** | | **~25 h** | **~$105** |
| Generation, SFT, engine tables, A2/A4/A5/A6/A9 | Local / free tiers | ~60 h | $0 |

Modal issues $30/month in credits, which covers reruns beyond the reserve. Frontier-model
calibration numbers are API-side and evaluated on a 1,000-position subset.

---

## 12. Risks and kill criteria

| Risk | Evidence it is real | Mitigation |
|---|---|---|
| **RL adds nothing over SFT** | 2507.00726: 1k o3 traces before RL gave no improvement; all runs plateaued at 25–30% | Gate at step 200. If M3 (move-only) is flat, that is itself informative — it means the grounding terms carry the effect, which is the thesis. Consider brief mid-training on chess text so RL has something to amplify. |
| **The verifier gets gamed** | Predictable: terse traces asserting only trivially-true facts | R_coverage, a claim-count floor, manual inspection of top-reward rollouts every 100 steps. This is the failure that looks like success on the curve. |
| **Claim extraction is noisy** | ACT-Eval: LLM decomposition "can strip necessary context". Confirmed here — a naive extractor produced 221 false violations in 3,000 traces (§5.1) | Fixed by section-aware scoring and line replay, pinned by tests. Still hand-validate on 200 traces before the gradient. |
| **Acceptance rate too low to build a corpus** | Teacher scores 0.0% on board-state comprehension | Measured in week 1 on 1,000 positions. If under ~20%, change teacher — **never loosen the filter**, that is the method. |
| **Getting scooped** | Toronto shipped twice in ten months; ACT-Eval's future work is "expand the tool suite" — one step from using it as a reward | Move fast on §4–5; lean on the two axes neither group can trivially add (strategic positions, faithfulness). Preprint early. |
| **Near-tie band has no reward variance** | Group-relative advantage collapses when all rollouts score alike | Measure within-group σ per band before committing to the Set B mix; require σ > 0.05. |

**Hard kill criteria, in order:**

1. **Week 1** — 1,000 prompts generated in one offline batch with no HTTP. Nothing downstream matters otherwise.
2. **Week 3** — extractor agreement with hand labels ≥ 90%, else R_precision is noise.
3. **Week 6** — verified-SFT must beat leaked-SFT (M1) on held-out move quality, else the §1.2 diagnosis is wrong.
4. **Week 8** — M6 must beat M3 on claim precision without losing move quality. If both are flat, the verifier is being gamed; if precision rises while quality falls, reweight. If neither resolves, pivot to the data + faithfulness contribution and drop the RL claim.

---

## 13. Timeline

| Week | Work | Gate |
|---|---|---|
| **1** | Offline-batch generation path, local and Modal. Fix the F8 parser bug. Measure real throughput against §11.3. | 1,000 prompts, one shot, no HTTP |
| **2–3** | Build the verifier **before** the model. Validate against the existing 10,296 answers (occupancy answer already known: 8.3%). Hand-label 200 traces. | Extractor ≥90% agreement |
| **3–4** | Full engine tables. Source and band Set B. Freeze Set D with contamination controls built in, not bolted on. | Within-band reward σ > 0.05 |
| **4–5** | **Full generation pass over Sets A and B**, k=1 then k=4 on rejects. | Acceptance ≥20%; else change teacher |
| **6** | SFT: M6 arm, M1 leaked arm, M2 discovery-unverified arm. A2 ablation falls out for free. | Verified arm beats leaked arm |
| **7–9** | GRPO: M6 (2 seeds), M3, M4, M2, A1/A3. Inspect top-reward rollouts every 100 steps. | M6 > M3 on claim precision at no quality cost |
| **10** | Verified reranking (A5), faithfulness probes, A4/A6/A8/A9. | Faithfulness–accuracy correlation positive and significant |
| **11** | Full evaluation, all five tiers, Elo runs, 200-failure error analysis. | — |
| **12** | Write-up, release verifier package + datasets + configs, preprint. | — |

---

## 14. Reproducibility

Journal submissions are increasingly gated on this; treat it as a deliverable, not an afterthought.

- **Released:** verifier package (pip-installable, CPU-only), Sets A–D with engine tables, all
  training configs, all decoding configs, LoRA adapters, the evaluation harness, and the exact
  Stockfish version and node limits.
- **Determinism:** node limits rather than time limits everywhere; fixed seeds recorded per run;
  reward computation is pure-function and testable.
- **Single-GPU claim:** every experiment reproducible on one 80GB device; the full data pipeline on
  one 24GB device. Include a `make reproduce` path that regenerates the headline table.
- **Datacard** for each set: provenance, licence, band distribution, contamination controls, and the
  §1.2 defect list for Set C so its role as a negative control is unambiguous.

---

## 15. Appendix

### 15.1 Audit provenance

Computed with `python-chess` 1.11.2 against `GRPO_GM_dataset.csv` (150,000 rows) and
`10k_chunk_1.csv` (10,296 rows).

| Measurement | Sample | Seed |
|---|---|---|
| FEN validity, legality, phase, side-to-move | 20,000 positions | 0 |
| Board-grounding claim checks | 3,000 final answers | 0 |
| Engine-line structure, cp gaps | 30,000 prompts | 1 |
| Annotation mutual information | 40,000 positions | 3 |
| Template counts, pawn-skeleton clustering, trace markers, n-grams | full corpus | — |

**Not verified.** No model has been trained, no engine run, no generation performed. Throughput
figures in §11.3 are derived from KV-cache arithmetic and are to be confirmed in week 1.

### 15.2 Known bugs in `Annotation_generator/annotator.py`

1. `_get_top_lines` opens and closes a Stockfish process per position — use a persistent pool.
2. `Limit(time=2.0, depth=30)` — switch to a node limit for reproducibility.
3. `main()` still contains `df.sample(n=5)`.
4. `extracted_reasoning` parser fails silently in 99.8% of rows (F8) — root-cause before regenerating.
5. Output path `GRPO_variant/Annotation_generator/lichess_reasonings.csv` is hardcoded and relative.

### 15.3 Naming

Do not describe the MATE annotation field as a "Grandmaster annotation" anywhere in the paper or the
datacard. It is a rule-based template label drawn from a 200-item set (F2), and the MATE paper itself
describes it as such.
