# Simple Chatbot

A learning project demonstrating rule-based chatbot implementations in Python, progressing from basic dictionary lookup to regex pattern matching with conversation context.

## Files

### `chatbot.py`
A minimal chatbot using a dictionary to map keywords to responses. Matches user input against predefined keys via substring search.

### `enhanced_chatbot.py`
An improved version that adds:
- **Regex pattern matching** for flexible input recognition
- **Randomized responses** for variety
- **Conversation context** — remembers the user's name and tracks interaction count
- **Help command** for discoverability
- **Built-in features** like current time and jokes

## Usage

```bash
# Basic version
python chatbot.py

# Enhanced version
python enhanced_chatbot.py
```

Type messages to chat. Type `bye` to exit, or `help` (enhanced version) for a list of capabilities.

## Requirements

Python 3.x (standard library only, no external dependencies).
