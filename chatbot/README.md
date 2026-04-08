# ML Chatbot

An intent-classification chatbot built with TensorFlow and NLTK. It recognizes user intents (greetings, goodbyes, help requests, etc.) using a neural network and responds with predefined answers.

## How It Works

1. User messages are tokenized, lemmatized, and stripped of stopwords using NLTK
2. A neural network classifies the preprocessed text into one of several intents
3. If the confidence score exceeds a threshold, the bot picks a random response from that intent's response pool; otherwise it asks the user to rephrase

## Project Structure

| File | Description |
|---|---|
| `ml_chatbot.py` | Standalone chatbot using a CNN (Conv1D + Embedding). Trains on inline data and runs an interactive chat loop. |
| `improved_ml_chatbot.py` | Enhanced version using a Bidirectional LSTM. Loads intents from `training_data.json`, saves/loads the trained model (`chatbot_model.h5`), and supports a debug mode. |
| `add_more_intents.py` | Utility to add new intents (about, capabilities, weather) to `training_data.json` without overwriting existing data. |
| `debug_chatbot.py` | Sets the `CHATBOT_DEBUG` environment variable so the improved chatbot prints intent predictions and confidence scores. |
| `requirements.txt` | Python dependencies. |

## Setup

### Prerequisites

- Python 3.11+

### Installation

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic chatbot (CNN model)

```bash
python3 ml_chatbot.py
```

Trains a 1D CNN on built-in intent data, then starts an interactive chat. Type `quit` to exit.

### Improved chatbot (Bi-LSTM model)

```bash
# (Optional) Add extra intents first
python3 add_more_intents.py

# Run the chatbot
python3 improved_ml_chatbot.py
```

On first run it trains a Bi-LSTM model and saves it to `chatbot_model.h5`. Subsequent runs load the saved model and skip training.

### Debug mode

To see intent predictions and confidence scores while chatting:

```bash
CHATBOT_DEBUG=True python3 improved_ml_chatbot.py
```

Or run `debug_chatbot.py` before launching the chatbot in the same process.

## Model Architectures

### `ml_chatbot.py` -- CNN

```
Embedding (128-dim) -> Dropout(0.2) -> Conv1D(64, kernel=3) -> GlobalMaxPooling1D -> Dense(64, relu) -> Dropout(0.3) -> Softmax
```

Includes data augmentation (lowercase variants, simulated typos). Uses a 0.4 confidence threshold.

### `improved_ml_chatbot.py` -- Bi-LSTM

```
Embedding (100-dim) -> Bi-LSTM(64) -> Dropout(0.2) -> Bi-LSTM(32) -> Dropout(0.2) -> Dense(64, relu) -> Dropout(0.2) -> Dense(32, relu) -> Softmax
```

Uses a 0.5 confidence threshold. Tracks conversation history (last 10 exchanges).

## Adding Custom Intents

Edit `training_data.json` directly or use `add_more_intents.py` as a template. Each intent needs:

```json
{
    "tag": "intent_name",
    "patterns": ["example input 1", "example input 2"],
    "responses": ["bot response 1", "bot response 2"]
}
```

After modifying intents, delete `chatbot_model.h5` so the improved chatbot retrains on the updated data.

## Built-in Intents

| Intent | Examples |
|---|---|
| greeting | "Hi", "Hello", "Good morning" |
| goodbye | "Bye", "See you later", "Take care" |
| thanks | "Thank you", "Thanks a lot" |
| help | "Help", "What can you do" |
| capabilities | "Tell me about yourself", "Who are you" |
| fallback | Unrecognized topics (math, weather, etc.) |

## Dependencies

- **TensorFlow 2.15** -- model training and inference
- **NumPy 1.24** -- numerical operations
- **NLTK 3.9** -- tokenization, stopword removal, lemmatization
- **scikit-learn 1.6** -- train/test splitting
