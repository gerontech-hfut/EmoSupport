# AnnoMI (Expert-Annotated Motivational Interviewing Dialogues)

## Overview
AnnoMI is a public dataset of **133** professionally transcribed and **expert-annotated** counselling dialogues demonstrating **high- and low-quality Motivational Interviewing (MI)**. It is intended to support NLP research on counselling/therapy conversations and MI-related behaviors.

## Official Source (Data & Code)
- GitHub repository: https://github.com/uccollab/AnnoMI

## What’s Included (in the official repo)
Two CSV releases are provided:
1. **`AnnoMI-simple.csv`**: a simplified version with high-level MI annotations.
2. **`AnnoMI-full.csv`**: an extended version that includes additional expert-annotated attributes per utterance.

## Data Format (high-level)
Each row corresponds to an utterance with transcript/session metadata. Key columns include (see repo README for the full list):
- `transcript_id`, `mi_quality` (high/low), `topic`, `utterance_id`, `interlocutor` (therapist/client), `timestamp`, `utterance_text`
- High-level annotation examples: `main_therapist_behaviour` (e.g., reflection/question/therapist_input/other), `client_talk_type` (e.g., change/neutral/sustain)
- Full version additionally includes finer-grained flags/subtypes (e.g., whether therapist input/reflection/question exists and their subtypes) and `annotator_id`.