# LoRA Adapter for the Supporter Model

## Overview

This repository provides the LoRA adapter weights for a **supporter model** designed for supportive response generation in emotional support dialogues.  
The adapter was fine-tuned on top of **Llama 3.1 8B Instruct** and is intended to specialize the base model for the supporter-side generation task.

Rather than releasing a fully merged checkpoint, this repository distributes the **parameter-efficient LoRA adapter**, which should be loaded together with the corresponding base model during inference.

## Download

The LoRA adapter weights can be downloaded from the following Google Drive link:

**📥 LoRA Adapter Download:** https://drive.google.com/drive/folders/1jmUvTZN6kmodUy-RN92cnqGRBSRY_0Ce?usp=drive_link

## Base Model

This adapter was trained on:

- **Base model:** Llama 3.1 8B

To use this adapter correctly, please load it on top of the same base model.

## Intended Use

The adapter is intended for research and experimental use in:

- emotional support dialogue generation,
- supportive response modeling,
- dialogue system personalization and behavior adaptation.

It is primarily designed to generate supporter responses conditioned on the dialogue context.  
As with other large language models, outputs may be imperfect, biased, or contextually inappropriate in some cases; human review is recommended for sensitive applications.

## Files

The main files in this repository are:

- `adapter_model.safetensors`: LoRA adapter weights
- `adapter_config.json`: LoRA configuration
- `README.md`: model card and usage instructions

## Inference Example

Below is a minimal example showing how to load the base model and the LoRA adapter for inference.  
To match the training-time prompting format more closely, the example uses the same supporter prompt construction function.

```python
import json
from typing import List, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def sq_escape(text: str) -> str:
    """A minimal escaping utility for safely inserting user text into the prompt."""
    return text.replace("\\", "\\\\").replace("'", "\\'")


def build_supporter_prompt(
    dialog_history_list: List[str],
    query_speaker: str,
    query_content: str,
    profile_seeker: Optional[Any] = None,
) -> str:
    if profile_seeker is None:
        seeker_profile_str = "None"
    else:
        try:
            seeker_profile_str = json.dumps(profile_seeker, ensure_ascii=False)
        except Exception:
            seeker_profile_str = str(profile_seeker)

    try:
        dialog_history_str = json.dumps(dialog_history_list, ensure_ascii=False)
    except Exception:
        dialog_history_str = str(dialog_history_list)

    return (
        "You are a supporter helping a seeker with emotional difficulties, aiming to reduce the seeker's emotional distress through dialogue.\n"
        "Select the most appropriate strategy from the following eight strategies based on the dialog history and the current query.\n"
        "Eight Strategies:\n"
        "1. Question: Asking for information related to the problem to help the help-seeker articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get specific information.\n"
        "2. Restatement or Paraphrasing: A simple, more concise rephrasing of the help-seeker's statements that could help them see their situation more clearly.\n"
        "3. Reflection of Feelings: Articulate and describe the help-seeker's feelings.\n"
        "4. Self-disclosure: Divulge similar experiences that you have had or emotions that you share with the help-seeker to express your empathy.\n"
        "5. Affirmation and Reassurance: Affirm the help-seeker's strengths, motivation, and capabilities and provide reassurance and encouragement.\n"
        "6. Providing Suggestions: Provide suggestions about how to change, but be careful to not overstep and tell them what to do.\n"
        "7. Information: Provide useful information to the help-seeker, for example with data, facts, opinions, resources, or by answering questions.\n"
        "8. Others: Exchange pleasantries and use other support strategies that do not fall into the above categories.\n"
        "Keep replies brief without additional pronouns or extra elements.\n"
        f"Seeker profile (nullable): {seeker_profile_str}; "
        f"Dialog history: {dialog_history_str}; "
        f"What seeker says to you: {{'speaker': '{query_speaker}', 'content': '{sq_escape(query_content)}'}}"
    )


# Replace with the actual base model path or Hugging Face model ID
base_model_name = "meta-llama/Llama-3.1-8B"

# Replace with the local path or repository ID of this LoRA adapter
lora_path = "path/to/your-lora-adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,   # or torch.float16 depending on your hardware
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# Example input
dialog_history_list = [
    "Seeker: I have been under a lot of pressure at work recently.",
    "Supporter: That sounds exhausting. Would you like to tell me what has been weighing on you the most?"
]
query_speaker = "seeker"
query_content = "I feel like I am falling behind and cannot relax even at home."
profile_seeker = {
    "emotion": "stress",
    "situation": "work overload"
}

prompt = build_supporter_prompt(
    dialog_history_list=dialog_history_list,
    query_speaker=query_speaker,
    query_content=query_content,
    profile_seeker=profile_seeker,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
