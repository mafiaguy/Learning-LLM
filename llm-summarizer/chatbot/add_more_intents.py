import json

# Load existing training data
try:
    with open('training_data.json', 'r') as file:
        training_data = json.load(file)
except FileNotFoundError:
    # Create new training data if file doesn't exist
    training_data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": [
                    "Hi", "Hello", "Hey", "Good morning",
                    "Hi there", "Greetings", "How are you", "What's up"
                ],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! How may I assist you?"
                ]
            },
            {
                "tag": "goodbye",
                "patterns": [
                    "Bye", "See you later", "Goodbye", "Take care",
                    "I'm leaving", "See ya", "Later", "I have to go"
                ],
                "responses": [
                    "Goodbye! Have a great day!",
                    "Bye! Come back soon.",
                    "Take care! Hope to see you again."
                ]
            }
        ]
    }
# Add new intents
new_intents = [
    {
        "tag": "about",
        "patterns": [
            "Who are you", "What are you", "Tell me about yourself",
            "What is your purpose", "How do you work"
        ],
        "responses": [
            "I'm a simple machine learning chatbot designed to demonstrate NLP concepts.",
            "I'm an AI assistant created to help answer questions and demonstrate chatbot functionality.",
            "I'm a chatbot built with Python, TensorFlow, and NLTK to showcase how machine learning can be used for conversations."
        ]
    },
    {
        "tag": "capabilities",
        "patterns": [
            "What can you do", "What are your features", "How can you help me",
            "What are your capabilities", "Tell me what you do"
        ],
        "responses": [
            "I can answer questions, provide information, and have simple conversations on topics I've been trained on.",
            "I can understand different ways of asking questions and try to provide helpful responses.",
            "I'm designed to recognize the intent behind your messages and respond appropriately."
        ]
    },
    {
        "tag": "weather",
        "patterns": [
            "What's the weather like", "Is it going to rain today", "Temperature today",
            "Weather forecast", "Is it sunny outside"
        ],
        "responses": [
            "I'm sorry, I don't have access to real-time weather data. You would need to integrate an external weather API for that functionality.",
            "As a simple demo chatbot, I can't check the weather. That would require connecting to a weather service API.",
            "I don't have the capability to check weather conditions. That requires access to external data sources."
        ]
    }
]
# Add the new intents to existing training data
for new_intent in new_intents:
    # Check if intent already exists
    intent_exists = False
    for existing_intent in training_data["intents"]:
        if existing_intent["tag"] == new_intent["tag"]:
            # Merge patterns and responses if intent exists
            existing_intent["patterns"].extend(new_intent["patterns"])
            existing_intent["responses"].extend(new_intent["responses"])
            intent_exists = True
            break

    # Add as new intent if it doesn't exist
    if not intent_exists:
        training_data["intents"].append(new_intent)

# Save the updated training data
with open('training_data.json', 'w') as file:
    json.dump(training_data, file, indent=4)

print("Training data updated and saved to 'training_data.json'")
