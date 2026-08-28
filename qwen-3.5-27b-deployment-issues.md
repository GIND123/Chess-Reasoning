# Qwen3.5-27B Chess Reasoning Annotation — Project Notes

> **Purpose of this document**: This is a consolidated log of the chess-reasoning synthetic annotation project — dataset construction, model benchmarking, inference optimisation, and large-scale deployment attempts. It merges `Chess annotation.md`, `Benchmark of LLM Annotation.md`, and `Llama.cpp optimisation.md` into the original deployment-issues log so a future working session has full context in one place without needing to cross-reference separate notes.

---

## 0. Project Goal

Generate a large-scale (target: **100,000 samples**) synthetic chess-reasoning annotation dataset by running an LLM over FEN positions + engine analysis + GM-style annotations, in order to produce training data (e.g. for GRPO). The pipeline is: **source data → prompt construction → LLM reasoning generation at scale → benchmarking to pick the best generator model → optimised local inference**. The core unresolved blocker throughout is **scaling generation to 100k samples**, which every deployment approach tried so far has failed to solve (see Section 4).

---

## 1. Dataset Construction

### 1.1 Early attempts (scrapped)

- Initial data was **2k entries of an annotated GM game dataset** taken from [chess.com](https://www.chess.com/home), plus **100k non-annotated entries** cleaned from Lichess.
- The [Lichess dataset](https://database.lichess.org/) contains raw FEN strings and best moves; best moves were verified using **Stockfish 17**, capped at 2 seconds per position.
- **Outcome: scrapped.** The chess.com dataset's annotations were broken/incomplete even after regex cleanup, and were not usable as gold-standard annotations.

### 1.2 MATE dataset (current source)

- Switched to the [MATE dataset](https://huggingface.co/datasets/OutFlankShu/MATE_DATASET), which contains FEN, best move, and annotation together, and tests whether an LLM can pick the best move from a set of options plus justify it.
- Created for the paper *[Explore the Reasoning Capability of LLMs in the Chess Testbed](https://arxiv.org/abs/2411.06655)*, co-authored by [Hou Yifan](https://en.wikipedia.org/wiki/Hou_Yifan) (chess grandmaster and neuroscientist, Peking University).
- MATE has **300k+ entries**; Stockfish was used to select which annotation/best-move pair to keep, yielding **220k usable annotation + best-move pairs**.
- Sample MATE entry:
  ```
  The FEN of the given chess board is "4k2r/p4ppp/2p3P1/2b1P2n/8/P1N5/1P1rQ1PP/R4R1K b k - 0 20". Which move is better? MoveA:d2e2, Trade a lesser piece for a higher-value piece to gain a positional advantage. TacticA: d2e2 g6f7 e8d7 c3e2 Trade the lower value piece for a higher value piece. MoveB:h5g3, Relocate the piece to a dynamic square, it's more influential on the board. TacticB: h5g3 h2g3 Trade the lower value piece for a higher value piece.
  ```
- MATE's own annotations are too simple for the target use case, motivating the creation of a new, more detailed reasoning dataset (below).

### 1.3 Prompt generation

- MATE was reduced to **150k entries** (chosen as a reasonable size for GRPO training).
- A Python script converts each raw MATE entry into a standardised prompt, adding:
  - The full ASCII board rendering (not just FEN), to aid spatial reasoning.
  - Guidance based on *[Mind's Eye of LLMs: visualization-of-thought elicits spatial reasoning in large language models](https://arxiv.org/abs/2404.03622)*.
- **Prompt template:**
  ```python
  """Given a board's FEN string: 
  <FEN>

  The ASCII board for the given FEN string is:
  +---+---+---+---+---+---+---+---+
  | . | . | . | . | . | . | . | . |
  +---+---+---+---+---+---+---+---+
  ... (8x8 grid) ...

  Use the following annotation provided by a Grandmaster to help guide your reasoning only when viable. 
  Annotation : Position the piece more actively, giving it greater board command.

  Use the below Centipawn loss (Cp) and move sequence to guide your reasoning:

  Line 1; <CP score/ Mate>: <Line 1>
  Line 2; <CP score/ Mate>: <Line 2>
  Line 3; <CP score/ Mate>: <Line 3>

  The best move is : <Best Move>

  Give reasoning explaining why this is the best move basing your answer on the given information."""
  ```
- **Filled example (from the prompt-generation code):**
  ```python
  USR_PROMPT = """Given a board's FEN string: 
  8/R7/p2p2p1/P2Bp3/4Pk1p/1P2r3/8/7K b - - 2 50
  ...
  Annotation : Position the piece more actively, giving it greater board command.

  Line 1; Mate(3): f4g3 a7f7 e3e1 f7f1 e1f1
  Line 2; Cp(0): e3e2 d5c4 e2b2 a7g7 g6g5 c4a6 b2b1 h1h2 b1b2 h2g1 g5g4 a6c8 b2b1 g1f2 g4g3 f2g2 b1b2 g2f1 b2b1 f1g2
  Line 3; Cp(-65): e3h3

  The best move is : f4g3
  ...
  """
  ```
- These 150k filled prompts (FEN + best move, no model-generated reasoning yet) were saved to CSV — this is the **base, unannotated prompt set** referenced in Section 4's dataset table (`GRPO_GM_dataset.csv`).

### 1.4 Sample of the unannotated prompt set (real record)

The example below is an actual rendered row from the base prompt CSV (this is what gets fed to the LLM for reasoning generation — the "Annotation" line here is the GM-sourced hint carried over from MATE, not model output):

```
Given a board's FEN string: 
r1b2rk1/pp3ppp/2nbpn2/8/3P4/4BN2/PqB1QPPP/RN3RK1 w - - 0 12

The ASCII board for the given FEN string is:
+---+---+---+---+---+---+---+---+
| r | . | b | . | . | r | k | . |
+---+---+---+---+---+---+---+---+
| p | p | . | . | . | p | p | p |
+---+---+---+---+---+---+---+---+
| . | . | n | b | p | n | . | . |
+---+---+---+---+---+---+---+---+
| . | . | . | . | . | . | . | . |
+---+---+---+---+---+---+---+---+
| . | . | . | P | . | . | . | . |
+---+---+---+---+---+---+---+---+
| . | . | . | . | B | N | . | . |
+---+---+---+---+---+---+---+---+
| P | q | B | . | Q | P | P | P |
+---+---+---+---+---+---+---+---+
| R | N | . | . | . | R | K | . |
+---+---+---+---+---+---+---+---+

Use the following annotation provided by a Grandmaster to help guide your reasoning only when viable. 
Annotation : Sacrifice a piece to unlock a file or diagonal in proximity to the opposing king

Use the below Centipawn loss (Cp) and move sequence to guide your reasoning:

Line 1; Cp(521): c2h7 f6h7 e2b2 b7b6 b1a3 a3b5 a6b5 b2b5 c6b4 e3d2 a7a5 g1h1 h7f6 b5e2 a8c8 g2g3 f6d5
Line 2; Cp(104): d4d5 b2a1 d5c6 e6e5 e2d2 e5e4 b1c3 a1b2 f1b1 b2a3 c3b5 a3a2 b5d6 b7c6 f3g5 c8a6 g5e4 f6e4
Line 3; Cp(-92): f1c1 b2a1 b1a3 a1c1 e3c1 d6e7 a3c4 c8d7 c1g5 c6b4 c4e5 f8c8 c2b3 d7e6 h2e5 g5f6 e7e5 g5f6 e5

The best move is : c2h7

Give reasoning explaining why this is the best move basing your answer on the given information.
```

### 1.5 Reasoning generation (LLM call)

- Each of the 150k prompts is passed to an LLM to generate the actual free-text reasoning annotation.
- **System prompt used:**
  ```python
  SYS_PROMPT = """You are a professional chess reasoning agent. Respond in a single concise paragraph with no bullet points or emotes. Do not mention centipawn scores or any provided evaluation data; use them only internally. Use the ASCII board to justify why the move is strong. Provide a clear explanation of the best move with both tactical and positional insights. Format the final output as: "Reasoning": <reason>."""
  ```
- **Model chosen for generation:** unsloth's quantised **Qwen3.5-27B-Q5_K_M** ([HF link](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF)) — selected based on the benchmarking results in Section 2.
- Run via **llama.cpp** (see Section 3) to allow maximum control over execution and performance, and to allow execution across Colab, Modal, etc.

---

## 2. Model Benchmarking & Selection

### 2.1 Models evaluated

Five open-source **reasoning models** were benchmarked (reasoning models were chosen for their stronger multi-turn/complex-task accuracy, relevant to chess reasoning):

| Model | Quant | HF Link |
|---|---|---|
| Qwen3-30B-A3B-Thinking-2507 (unsloth GGUF) | Q8_0 | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF) |
| Phi-4-reasoning-plus (unsloth GGUF) | Q8_0 | [link](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF) |
| Qwen3-30B-A3B-Thinking-2507 (unsloth GGUF) | UD-Q5_K_XL | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF) |
| Qwen3-32B (unsloth GGUF) | UD-Q4_K_XL | [link](https://huggingface.co/unsloth/Qwen3-32B-GGUF) |
| Qwen3.5-27B (unsloth GGUF) | Q5_K_M | [link](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) |

> Note: "DeepSeek" appears as a fifth entry in the results charts below — treat it as the fifth benchmarked model alongside Phi4-Reasoning, Qwen3.5-27B, Qwen3-30B-A3B, and Qwen3-32B (the source note above lists the same GGUF link twice for the two Qwen3-30B-A3B rows — worth double-checking against the original benchmark script when picking this back up).

### 2.2 Methodology

- Each model was tested in **3 prompting settings**: **Base**, **One-Shot**, **Few-Shot**.
- All benchmarking was run on an **RTX 3090 (24GB VRAM)** system, using **Ollama** for inference (Ollama was fine for this smaller-scale benchmarking loop, unlike the 100k-scale generation run — see Section 4).
- Outputs were scored with an **LLM-as-a-Judge** methodology (per *[A survey on LLM-as-a-Judge](https://www.cell.com/the-innovation/fulltext/S2666-6758(25)00456-4)*), on 3 criteria (a 4th, "overall", is derived/reported alongside):
  - **reasoning_score** — how well the model's reasoning matches the gold explanation (1–10)
  - **concept_score** — how well the model understands the chess concept/theme (1–10)
  - **annotation_score** — how useful the output is as a training annotation (1–10)
  - **overall_score** — reported in the results as "Overall"
- **Judge prompt:**
  ```python
  JUDGE_PROMPT = """You are a chess expert. Score the model's reasoning against the gold standard.

  FEN: {fen}

  GOLD STANDARD:
  - Best Move: {gold_move}
  - Annotation: {gold_annotation}
  - Explanation: {gold_explanation}
  - Theme: {gold_theme}

  MODEL'S REASONING:
  {model_answer}

  Give three scores from 1-10:
  - reasoning_score: How well does the model's reasoning match the gold explanation? (10=perfect alignment, 1=completely wrong)
  - concept_score: How well does the model understand the chess concept/theme? (10=deep understanding, 1=no understanding)
  - annotation_score: How useful is this as a training annotation? (10=excellent, 1=useless)

  Respond ONLY with this JSON:
  {{
  "reasoning_score": <1-10>,
  "concept_score": <1-10>,
  "annotation_score": <1-10>,
  "overall_score": <1-10>
  }}"""
  ```
- **Judge models:** two judges were used to reduce single-judge bias — **Gemini 3 Flash Preview** and **GPT-OSS 120B**, both accessed via **Ollama Cloud (free tier)**.

### 2.3 Results — GPT-OSS 120B as judge

**Base prompting**

| Model | Reasoning | Concept | Annotation | Overall |
|---|---|---|---|---|
| DeepSeek | 3.9 | 4.2 | 3.4 | 3.8 |
| Phi4-Reasoning | 6.2 | 6.1 | 4.9 | 5.8 |
| **Qwen3.5-27B** | **8.2** | **7.9** | **7.1** | **7.9** |
| Qwen3-30B-A3B | 6.1 | 6.3 | 5.2 | 6.1 |
| Qwen3-32B | 6.0 | 6.2 | 5.2 | 5.8 |

**One-Shot prompting**

| Model | Reasoning | Concept | Annotation | Overall |
|---|---|---|---|---|
| DeepSeek | 4.9 | 5.0 | 3.9 | 4.6 |
| Phi4-Reasoning | 5.1 | 5.4 | 4.3 | 5.1 |
| **Qwen3.5-27B** | **7.6** | **7.4** | **6.4** | **7.3** |
| Qwen3-30B-A3B | 6.9 | 6.9 | 5.9 | 6.8 |
| Qwen3-32B | 5.6 | 5.9 | 4.7 | 5.5 |

**Few-Shot prompting**

| Model | Reasoning | Concept | Annotation | Overall |
|---|---|---|---|---|
| DeepSeek | 2.5 | 3.1 | 2.3 | 2.5 |
| Phi4-Reasoning | 6.0 | 5.9 | 4.7 | 5.6 |
| **Qwen3.5-27B** | **7.8** | **7.9** | **6.7** | **7.7** |
| Qwen3-30B-A3B | 6.1 | 5.9 | 5.3 | 5.8 |
| Qwen3-32B | 5.4 | 5.8 | 4.5 | 5.2 |

### 2.4 Results — Gemini 3 Flash Preview as judge

**Base prompting**

| Model | Reasoning | Concept | Annotation | Overall |
|---|---|---|---|---|
| DeepSeek | 2.8 | 3.7 | 2.3 | 3.0 |
| Phi4-Reasoning | 5.4 | 6.9 | 4.8 | 5.6 |
| **Qwen3.5-27B** | **8.6** | **9.2** | **8.5** | **8.7** |
| Qwen3-30B-A3B | 6.3 | 7.2 | 5.8 | 6.3 |
| Qwen3-32B | 4.4 | 5.3 | 3.7 | 4.4 |

**One-Shot / Few-Shot prompting:** not yet run with this judge — still open (see Section 5).

### 2.5 Analysis & conclusion

- **Qwen3.5-27B is the best-performing model across both judges and all three prompting settings**, consistently leading on reasoning, concept understanding, and annotation usefulness.
- **Base prompting outperforms one-shot and few-shot** for Qwen3.5-27B (e.g. under the GPT-OSS 120B judge: Base overall 7.9 vs. One-Shot 7.3 vs. Few-Shot 7.7) — extra in-context examples did not improve output quality in this setup.
- This is why **Qwen3.5-27B (Q5_K_M), used with base prompting,** was selected as the generator model for the full 150k reasoning-generation run (Section 1.5).
- Practical implication for infra: **Ollama cannot be well-optimised on the compute instances being used for the large run (Modal, Colab, Kaggle)**, so **llama.cpp** was adopted for the actual generation pipeline instead of Ollama (Section 3). Llama.cpp requires manual optimisation/compilation but gives the fastest, most-optimised local inference.

---

## 3. Inference Optimisation (llama.cpp)

### 3.1 Installation

- `pip install llama-cpp-python` was **not used** — it doesn't ship the latest build needed to run Qwen3.5-27B.
- Instead, [llama.cpp](https://github.com/ggml-org/llama.cpp) was built from source with CUDA GPU flags:
  ```bash
  cmake -B build -DGGML_CUDA=ON
  cmake --build build --config Release -j 8
  # higher number after -j allows for more threads used in processing, ergo faster compilation
  ```
- Once built, the model runs as a local server; inference is done by sending POST/GET requests to the server's port from Python.

### 3.2 Inference server config

```bash
llama.cpp/build/bin/llama-server \
-m models/Qwen3.5-27B-Q5_K_M/Qwen3.5-27B-Q5_K_M.gguf \
--n-gpu-layers -1 \
--ctx-size 16384 \
--batch-size 2048 \
--ubatch-size 512 \
--flash-attn on \
--cache-type-k q8_0 \
--cache-type-v q8_0 \
--parallel 4 \
--cont-batching \
--mlock \
--port 8081
```

| Flag | Purpose |
|---|---|
| `--n-gpu-layers -1` | Loads all model layers into GPU memory |
| `--ctx-size 16384` | Context length |
| `--cache-type-k q8_0`, `--cache-type-v q8_0` | Speeds up compute by caching KV to 8-bit quantisation |
| `--flash-attn on` | Enables flash attention (not available on all GPU architectures) |
| `--parallel 4` | Computes 4 instances in parallel — on a single GPU this gives only marginal speedup since the CPU processes the 4 prompts in parallel |
| `--batch-size 2048`, `--ubatch-size 512` | `batch-size` controls throughput and VRAM usage; `ubatch-size` controls GPU utilisation |

This server setup is the target inference config for the 100k-scale run, but see Section 4.4 — getting a robust **client ↔ server** pipeline around it at scale is still an open problem.

---

## 4. Deployment & Scaling Issues (100k-sample run)

This section logs every approach tried to run Qwen3.5-27B at scale (target: 100k samples) and why each one failed or fell short.

### 4.1 Ollama (Local, Quantized Model)

- Initial runs used a quantized variant of the model served via Ollama.
- Ollama only supports a single inference pass at a time (no parallel request handling).
- Result: only a small fraction of the required 100k samples could be generated in a reasonable timeframe.

### 4.2 Google Colab (A100)

- Attempted on a Google Colab A100 instance.
- CUDA was not pre-installed and had to be built from scratch on each session.
- The CUDA build process was unreliable — it would frequently crash outright, or even when the build succeeded, Ollama still failed to run afterward.
- Result: this approach did not produce a working setup.

### 4.3 Ollama on Modal

- Ollama was deployed on Modal.
- Modal's setup did not support parallel requests, same limitation as local Ollama.
- Result: the deployment did not run successfully and produced a range of errors, none of which resulted in a stable pipeline.

### 4.4 vLLM (Separate Server and Client Instances)

- Tried a two-instance architecture: one instance running the vLLM server, a separate instance sending requests to it.
- Failed for unclear reasons, including:
  - The client instance failing to identify/discover the vLLM server.
  - Requests failing to POST to the server successfully.

### 4.5 Datasets produced so far

Hosted at `GOVINDFROM/ChessReasoningDataset` on Hugging Face:

| File | Size | Description |
|---|---|---|
| `GRPO_GM_dataset.csv` | 230 MB | Base prompt set passed as input to Qwen 3.5 27B. No annotations yet — just the prompt, best move, and FEN string per sample (this is the 150k-prompt set from Section 1.3). |
| `1500_samples.csv` | 50 MB | Output from splitting the workload across Colab, the 3090s, and Modal. Only the 3090 runs actually produced usable results — 1,500 annotated samples. |
| `10k_chunk_1.csv` | 325 MB | 10k annotated chunks, generated on a 3090. |

### 4.6 Summary of blockers

| Approach | Environment | Key Blocker |
|---|---|---|
| Ollama (quantized) | Local | No parallel requests; throughput far below 100k sample requirement |
| Ollama | Google Colab (A100) | CUDA had to be built manually; builds crashed or left Ollama non-functional |
| Ollama | Modal | No parallel request support; multiple unresolved errors |
| vLLM | Server + client (separate instances) | Client couldn't discover server; POST requests failed |

---

## 5. Open Items for Next Session

- **Primary blocker to solve:** a reliable, parallel-capable inference pipeline that can sustain generation of the remaining ~148.5k samples (150k prompts − ~11.5k annotated so far across the two partial runs). The llama.cpp server config in Section 3.2 is the intended engine — next step is likely wrapping it in a proper client/queueing setup (rather than retrying vLLM's server/client split, or Ollama, both of which have documented failures above).
- **Debug vLLM discovery/POST failures** if vLLM is revisited — no root cause was identified, worth checking networking/port-exposure and vLLM version compatibility.
- **Fill in missing benchmark cells:** One-Shot and Few-Shot results for the Gemini 3 Flash Preview judge (Section 2.4) were never run — only Base is populated.
- **Verify the model list** in Section 2.1 — the source benchmarking note links the same Qwen3-30B-A3B-Thinking-2507 GGUF for two different quantisation rows; confirm which quant was actually used for the "DeepSeek" row shown in the results charts, since no DeepSeek model is listed among the 5 models described in the methodology text.
- **Merge annotated chunks** (`1500_samples.csv` + `10k_chunk_1.csv`) once more chunks exist, and track total annotated progress toward the 150k target.
