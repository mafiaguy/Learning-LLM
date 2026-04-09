# LLM Learning

A hands-on learning project that progressively builds chatbots and NLP tools -- from simple rule-based bots all the way to running a real LLM locally. Each folder represents a step up in complexity. Built as part of an AI course by [Practical DevSecOps](https://www.practical-devsecops.com/).

## Learning Path

The projects are ordered from simplest to most advanced:

| # | Folder | Approach | Key Concept |
|---|--------|----------|-------------|
| 1 | `simple-chatbot/` | Rule-based (dictionary + regex) | Pattern matching, no ML |
| 2 | `chatbot/` | Neural network (CNN, Bi-LSTM) | Intent classification with TensorFlow |
| 3 | `llm-chatbot/` | Pre-trained LLM (TinyLlama 1.1B) | Text generation with Hugging Face Transformers |
| 4 | `llm-summarizer/` | Pre-trained LLM (Falconsai T5) | Summarization pipeline, file/URL input |
| 5 | `llm-chatbot-rag/` | RAG with Phi-3-mini + FAISS | Retrieval-Augmented Generation over local documents |
| 6 | `sentiment_attack/` | Adversarial ML (TextAttack + DistilBERT) | Adversarial text attacks on sentiment classifiers |

---

## 1. Simple Chatbot (`simple-chatbot/`)

A rule-based chatbot with zero dependencies beyond Python's standard library. Good starting point to understand the basic chatbot loop.

- **`chatbot.py`** -- Matches user input against a dictionary of keywords using substring search.
- **`enhanced_chatbot.py`** -- Adds regex pattern matching, randomized responses, conversation context (remembers your name), and a help command.

### Run

```bash
cd simple-chatbot

# Basic version
python3 chatbot.py

# Enhanced version
python3 enhanced_chatbot.py
```

Type messages to chat. Type `bye` to exit (`help` works in the enhanced version).

**Requirements:** Python 3.x (standard library only).

---

## 2. ML Chatbot (`chatbot/`)

An intent-classification chatbot that trains a neural network to recognize what the user means (greeting, goodbye, help request, etc.) and picks a response from a pool.

- **`ml_chatbot.py`** -- Trains a 1D CNN on inline intent data. Includes data augmentation (lowercase variants, simulated typos). Uses a 0.4 confidence threshold.
- **`improved_ml_chatbot.py`** -- Uses a Bidirectional LSTM. Loads intents from `training_data.json`, saves/loads the trained model (`chatbot_model.h5`), and supports a debug mode that shows predictions and confidence scores.
- **`add_more_intents.py`** -- Utility to add new intents (about, capabilities, weather) to `training_data.json` without overwriting existing data.
- **`debug_chatbot.py`** -- Sets the `CHATBOT_DEBUG` env var for the improved chatbot.

### Run

```bash
cd chatbot

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Option A: CNN chatbot (trains on built-in data, no setup needed)
python3 ml_chatbot.py

# Option B: Bi-LSTM chatbot
# (Optional) Seed extra intents first:
python3 add_more_intents.py
# Then run:
python3 improved_ml_chatbot.py

# To see prediction details:
CHATBOT_DEBUG=True python3 improved_ml_chatbot.py
```

Type `quit` to exit. On first run the Bi-LSTM version trains and saves `chatbot_model.h5`; subsequent runs load the saved model. Delete `chatbot_model.h5` to retrain after changing intents.

**Requirements:** Python 3.11+, TensorFlow 2.15, NumPy, NLTK, scikit-learn (see `requirements.txt`).

---

## 3. LLM Chatbot (`llm-chatbot/`)

A chatbot powered by [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), a small open-source LLM that runs locally. Unlike the previous projects that use canned responses, this one generates free-form text.

- **`llm-chatbot.py`** -- Loads TinyLlama via Hugging Face Transformers and runs an interactive chat loop.

### Run

```bash
cd llm-chatbot

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the chatbot (downloads ~2 GB model on first run)
python3 llm-chatbot.py
```

Type your message and press Enter. Type `X` to exit.

**Requirements:** Python 3.8+, PyTorch, Transformers, ~2 GB disk for the model. GPU recommended but CPU works (`device_map="auto"`).

---

## 4. LLM Summarizer (`llm-summarizer/`)

A text summarization tool using [Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization) (a fine-tuned T5 model). Demonstrates using an LLM for a specific task beyond chat.

- **`llm-summarizer.py`** -- Takes text input from the terminal and outputs a summary.
- **`new-llm-summarizer.py`** -- Enhanced version that can read input from a local file (`file://path`) or a URL (`http://`/`https://`).

### Run

```bash
cd llm-summarizer

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Basic summarizer (paste text directly)
python3 llm-summarizer.py

# Enhanced summarizer (supports file and URL input)
python3 new-llm-summarizer.py
```

Type or paste text to summarize. In the enhanced version, provide a `file:///path/to/file.txt` or `https://example.com/article` to summarize content from that source. Type `X` to exit.

**Requirements:** Python 3.8+, PyTorch, Transformers, ~1 GB disk for the model.

---

## 5. RAG Chatbot (`llm-chatbot-rag/`)

A Retrieval-Augmented Generation chatbot that answers questions using your own documents. Uses [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) as the LLM and FAISS for vector similarity search over document chunks.

- **`llm-chatbot-rag.py`** -- Loads text files from `documents/`, splits them into chunks, creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, stores them in a FAISS index, and retrieves relevant context to augment the LLM prompt.
- **`documents/`** -- Drop your text files here for the chatbot to use as its knowledge base.

### Run

```bash
cd llm-chatbot-rag

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your documents to the documents/ folder, then run
python3 llm-chatbot-rag.py
```

Type your question and press Enter. The chatbot retrieves relevant chunks from your documents and uses them as context for the answer. Type `X` to exit.

**Requirements:** Python 3.8+, PyTorch, Transformers, LangChain, FAISS, Sentence-Transformers, ~7 GB disk for the Phi-3-mini model. GPU recommended.

---

## 6. Sentiment Attack (`sentiment_attack/`)

An adversarial ML experiment that uses [TextAttack](https://github.com/QData/TextAttack) to craft adversarial examples against a DistilBERT sentiment classifier (`distilbert-base-uncased-finetuned-sst-2-english`). Demonstrates how small word substitutions can flip a model's prediction.

- **`sentiment_classifier.py`** -- Loads DistilBERT and exposes a `predict_sentiment()` function that returns label, confidence, and class probabilities.
- **`test_sentiment.py`** -- Runs the classifier on a set of sample texts to verify it works correctly.
- **`textattack_wrapper.py`** -- Wraps the sentiment classifier in TextAttack's `ModelWrapper` interface so it can be used as an attack target.
- **`run_attack.py`** -- Runs the TextFooler attack recipe on 20 examples, prints detailed results (original vs. perturbed text, confidence changes, word modification rates), and saves a summary to `attack_summary.txt` and `attack_results.csv`.

### Run

```bash
cd sentiment_attack

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test the sentiment classifier
python3 test_sentiment.py

# Run the adversarial attack
python3 run_attack.py
```

Results are printed to the terminal and saved to `attack_summary.txt`. The attack shows how synonym substitutions (e.g., "great" -> "outstanding") can flip sentiment predictions while preserving meaning.

**Requirements:** Python 3.11+, PyTorch, Transformers, TextAttack, TensorFlow, NLTK, scikit-learn (see `requirements.txt`). ~500 MB for the DistilBERT model.

---

## General Setup Notes

- Each project has its own `requirements.txt` -- create a separate virtual environment per folder to avoid dependency conflicts (especially between TensorFlow in `chatbot/` and PyTorch in the `llm-*` folders).
- The LLM-based projects download models from Hugging Face on first run. Ensure you have a stable internet connection.
- All projects run entirely locally -- no API keys or cloud services needed.
# llm-learning
