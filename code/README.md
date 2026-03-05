# Code Folder

This folder contains the runnable pipeline for generating and evaluating emotional-support dialogues on a persona-based test set.

## Files

### `prompt.py`
Defines the prompts used across the pipeline:

- **`prompt_supporter`**: Instructs the supporter model to select one of eight support strategies and output a structured JSON with a short chain-of-thought (4 steps) and the final reply.
- **`prompt_seeker`**: Instructs the seeker simulator / real seeker model to respond with exactly one sentence per turn, and to terminate by outputting `[END]` when sufficiently supported.
- **`prompt_perinf`**: Persona inference prompt to extract a structured seeker persona JSON (gender, age group, traits, communication style, occupation, roles, emotion, situation) with confidences.
- **`prompt_judge`**: ES-Skills scoring prompt to evaluate supporter responses (identification, comforting, suggestions, fluency, coherence, safety, overall) on a strict 1–5 scale.

### `main.py`
Implements the end-to-end evaluation loop on a test persona set:

- Loads persona profiles from `PERSONA_JSON`.
- Runs multi-turn conversations with:
  - a **Supporter** model (local OpenAI-compatible endpoint),
  - a **Real Seeker** model (OpenAI-compatible endpoint, e.g., GPT-5-mini),
  - a **Simulated Seeker** model (local endpoint used only for lookahead rollouts),
  - a **Persona Inference** model (local endpoint),
  - a **Valence** service (HTTP API) to score seeker messages.
- Performs **lookahead planning** after a few turns:
  - samples multiple supporter candidates at the root,
  - simulates seeker responses and valence scores,
  - selects the candidate maximizing expected valence improvement.
- Logs everything to `RUN_ROOT/`:
  - per-episode `history.jsonl` / `errors.jsonl`,
  - `final_history.json` and `summary.json`,
  - run-level `all_conversations.json` and `global_stats.json`.
- Supports resuming unfinished runs via checkpoint files.

## Training and Reproducibility

### Supporter Fine-Tuning

The supporter model used in this project is trained in two stages:

1. **Supervised Fine-Tuning (SFT)** for instruction-following adaptation on emotional-support dialogue data.
2. **Direct Preference Optimization (DPO)** for preference-based alignment using response pairs derived from dialogues.

### LoRA Configuration

For both SFT and DPO, we use LoRA-based adaptation with the following settings:

- LoRA rank: `r = 16`
- LoRA scaling: `alpha = 32`
- LoRA dropout: `0.1`

### Optimization Settings

All training runs use:

- Optimizer: **AdamW**
- Precision: **bf16**
- Number of epochs: **3**

For the DPO stage, we additionally use:

- DPO inverse temperature: `beta = 0.1`
- Learning rate: `5e-7`

### DPO Reference Model and Preference Filtering

The reference policy used in DPO, denoted as `\(\pi_{\mathrm{ref}}\)`, is the **SFT-initialized supporter model** with the same LoRA configuration. This reference model remains **frozen** throughout DPO training.

To improve preference quality and reduce label noise, preference pairs are filtered using a value-gap threshold:

- Threshold: `tau = 0.1`

Only pairs satisfying `|\Delta \hat{Q}| \ge tau` are retained for DPO optimization.

### Training Toolkit

All SFT and DPO experiments are implemented with **LlamaFactory**, an open-source framework for efficient fine-tuning of large language models and vision-language models. In our workflow, it is used as the primary training toolkit for LoRA-based supervised fine-tuning and preference optimization.

- GitHub: [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)

## Requirements

Python packages:
- `openai`
- `httpx`
- `tqdm`

External services (OpenAI-compatible or HTTP):
- Supporter endpoint (`BASE_URL_SUPPORTER`)
- Sim seeker endpoint for lookahead (`BASE_URL_SEEKER`)
- Persona inference endpoint (`BASE_URL_PERSONA`)
- Valence service (`VALENCE_BASE`)
- Real seeker endpoint (`BASE_URL_5mini`)

## Usage

1. Configure environment variables (recommended):
   - `PERSONA_SEGMENT` = `all|first|middle|last`
   - `BASE_URL_SUPPORTER`, `API_KEY_SUPPORTER`, `MODEL_SUPPORTER`
   - `BASE_URL_SEEKER`, `API_KEY_SEEKER`, `MODEL_SEEKER`
   - `BASE_URL_5mini`, `API_KEY_5mini`, `MODEL_5mini`, `REASONING_EFFORT_5MINI`, `VERBOSITY_5MINI`
   - `BASE_URL_PERSONA`, `API_KEY_PERSONA`, `MODEL_PERSONA`
   - `VALENCE_BASE`

2. Run:
   ```bash
   python main.py
