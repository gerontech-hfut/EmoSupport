# ESConv (Emotional Support Conversation)

## Overview
ESConv is an English **emotional support conversation (ESC)** dataset released with code by the CoAI group (Tsinghua University) for the ACL 2021 paper *Towards Emotional Support Dialog Systems*.  
It contains multi-turn dialogues between a help-seeker and a supporter, where the supporter turns are annotated with **support strategies**.

## Official Source (Data & Code)
- GitHub repository: https://github.com/thu-coai/Emotional-Support-Conversation

## What’s Included (in the official repo)
- `ESConv.json`: the main ESConv corpus (reported as **1,300 conversations** across **10 problem topics**).
- `FailedESConv.json`: additional **196 negative / failed** samples released later for future research.

## Data Format (high-level)
Each conversation is stored in JSON and typically includes:
- metadata such as `experience_type`, `emotion_type`, `problem_type`, `situation`
- post-conversation survey fields (if available)
- `dialog`: a list of turns, each turn containing `text`, `speaker` (e.g., seeker/supporter), and (for supporter turns) `strategy`

## Support Strategy Labels
The dataset uses **8** strategy categories, including:
- Questions
- Self-disclosure
- Affirmation and Reassurance
- Providing Suggestions
- Other
- Reflection of feelings
- Information
- Restatement or Paraphrasing
