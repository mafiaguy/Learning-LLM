import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
import os

# Check if debug mode is enabled
DEBUG_MODE = os.environ.get('CHATBOT_DEBUG', 'False').lower() in ('true', '1', 't')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Model parameters
max_words = 1000
max_len = 20
embedding_dim = 100
lstm_units = 64

# Text preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return ' '.join(tokens)

# Load training data from file
print("Loading training data...")
try:
    with open('training_data.json', 'r') as file:
        training_data = json.load(file)
except FileNotFoundError:
    print("Training data file not found. Using default training data.")
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

# Prepare text and labels from training data
texts = []
labels = []
responses = {}

for intent in training_data['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

# Preprocess all texts
processed_texts = [preprocess_text(text) for text in texts]

# Create tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(processed_texts)

# Save tokenizer for future use
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(processed_texts)

# Pad sequences
X = pad_sequences(sequences, maxlen=max_len)

# Convert labels to numerical format
unique_labels = list(set(labels))
label_dict = {label: i for i, label in enumerate(unique_labels)}
reverse_label_dict = {i: label for i, label in enumerate(unique_labels)}

# Save label dictionaries for future use
with open('labels.pickle', 'wb') as handle:
    pickle.dump((label_dict, reverse_label_dict), handle, protocol=pickle.HIGHEST_PROTOCOL)

y = np.array([label_dict[label] for label in labels])

# Check if model already exists
if os.path.exists('chatbot_model.h5'):
    print("Loading existing model...")
    model = load_model('chatbot_model.h5')
else:
    # Build the neural network model with dropout for better generalization
    print("Building new model...")
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(0.2),  # Add dropout to prevent overfitting
        Bidirectional(LSTM(lstm_units // 2)),
        Dropout(0.2),  # Add dropout to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.2),  # Add dropout to prevent overfitting
        Dense(32, activation='relu'),
        Dense(len(unique_labels), activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Increase epochs for better accuracy
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Save the model
    model.save('chatbot_model.h5')
    print("Model saved to 'chatbot_model.h5'")

# Function to predict intent from text
def predict_intent(text):
    # Preprocess input
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=max_len)

    # Get prediction
    prediction = model.predict(padded)[0]

    # Get top 3 predictions for debugging
    top_indices = prediction.argsort()[-3:][::-1]
    top_intents = [(reverse_label_dict[idx], prediction[idx]) for idx in top_indices]

    # Get best prediction
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    intent = reverse_label_dict[predicted_class]

    return intent, confidence, top_intents

# Function to generate response based on predicted intent
def get_response(intent, confidence, threshold=0.5):
    if confidence < threshold:
        return "I'm not sure I understand. Could you please rephrase that?", confidence

    return random.choice(responses[intent]), confidence

# Track conversation history for context
conversation_history = []

# Main chat loop with improvements
print("\nChatbot: Hello! How can I help you today? (type 'quit' to exit)")
if DEBUG_MODE:
    print("[Debug mode enabled - showing intent predictions and confidence scores]")

while True:
    # Get user input
    user_input = input("\nYou: ")

    # Exit condition
    if user_input.lower() == 'quit':
        print("\nChatbot: Goodbye! Have a great day!")
        break

    # Add to conversation history
    conversation_history.append(f"User: {user_input}")

    # Get intent and confidence
    intent, confidence, top_intents = predict_intent(user_input)

    # Show debug information if enabled
    if DEBUG_MODE:
        print("\n[DEBUG INFORMATION]")
        print(f"Processed input: '{preprocess_text(user_input)}'")
        print("Top predictions:")
        for idx, (intent_name, score) in enumerate(top_intents):
            print(f"  {idx+1}. {intent_name}: {score:.4f}")
        print(f"Selected intent: {intent} (confidence: {confidence:.4f})")

    # Get response
    response, conf = get_response(intent, confidence)

    # Add to conversation history
    conversation_history.append(f"Bot: {response}")

    # Print response
    if DEBUG_MODE:
        print(f"\nChatbot ({intent}, confidence: {confidence:.4f}): {response}")
    else:
        print(f"\nChatbot: {response}")

    # Keep history to last 10 exchanges to maintain context
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
