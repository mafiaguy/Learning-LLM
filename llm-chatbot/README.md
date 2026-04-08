# LLM Chatbot

A simple command-line chatbot powered by [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using Hugging Face Transformers.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python llm-chatbot.py
```

Type your message at the prompt and press Enter to get a response. Type `X` to exit.

## Requirements

- Python 3.8+
- ~2 GB disk space for model download (fetched automatically on first run)
- GPU recommended but not required (`device_map="auto"` will use CPU as fallback)
