# Persona Profiles Dataset

## Overview

This repository releases a set of **persona profiles** used in our paper:

This dataset is designed for studying **persona-aware emotional support conversation (ESC)** under **asymmetric information**, where the supporter must infer the seeker’s profile online from the dialogue history.

These persona profiles can be used for:

- dialogue synthesis
- multi-role supervised fine-tuning (SFT)
- simulated interactions
- preference data collection
- interactive evaluation

> Important: These persona profiles are **structured summaries** and do not include raw dialogue text.  
> If your pipeline references external corpora (e.g., ESConv, MESC, AnnoMI), please obtain those datasets from their official sources and follow their licenses.

---

## Files

Together, these files form the persona dataset and can be used for different experimental purposes, such as data synthesis, simulated interaction, preference collection, or evaluation.

---

## Persona Schema

Each persona entry is a JSON object with the following top-level structure (example):

```json
{
  "persona_index": 0,
  "macro_type": "Self-regulation & Behaviors",
  "problem_type": "Emotional Eating",
  "person_description": {
    "gender": { "value": "female|male|unknown" },
    "age_group": { "value": "teen|young|middle|elderly|unknown" },
    "traits": { "value": "free-text traits summary" },
    "communication_style": { "value": "free-text preferred communication style" },
    "occupation": {
      "major_group": { "value": "ISCO-08 major group name|unknown" },
      "specific_occupation": { "value": "free-text job title|unknown" }
    },
    "life_roles": { "value": ["Child", "Student", "Worker", "Parent", "..."] },
    "emotion_type": { "value": "free-text primary negative emotion|unknown" },
    "situation": { "value": "free-text situation description" }
  }
}