# emotion-recognition-chatbot
Clinical + conversational emotion classification using BERT, MELD, and DAIC-WOZ
# Emotion Recognition Chatbot

This project supports the development of a multimodal AI-driven virtual patient to train therapists in emotional awareness and communication.

## Goals

- Classify emotional tone from **text-only** inputs (phase 1)
- Fine-tune transformer models (e.g., BERT) on clinical datasets
- Benchmark model performance on complex clinical labels (e.g., trauma, guilt, resilience)

##  Structure
emotion-recognition-chatbot/
├── datasets/ # Raw and cleaned data
│ ├── meld/
│ └── daic-woz/
├── notebooks/ # EDA, modeling, evaluation
├── scripts/ # Training + preprocessing scripts
├── reports/ # Weekly reports + paper drafts
└── README.md

## Datasets Used

- [MELD](https://github.com/declare-lab/MELD)
- DAIC-WOZ
- EmpatheticDialogues (optional)
- IEMOCAP (upon access)

## Quick Start

1. Clone the repo  
2. Add datasets to `datasets/`
3. Run notebooks in `notebooks/`

## Team
- Prof. Akram Bayat (Supervisor)  
- Xinyan Liu  
- Shlok Bansal


