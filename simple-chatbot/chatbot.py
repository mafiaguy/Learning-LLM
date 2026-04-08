# Dictionary mapping user inputs to bot responses
responses = {
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I'm just a program, but I'm functioning as expected!",
    "what is your name": "I'm SimpleBot, a rule-based chatbot.",
    "what can you do": "I can answer simple questions based on predefined patterns.",
    "bye": "Goodbye! Have a great day!",
}

# Display welcome message
print("="*50)
print("Simple Rule-Based Chatbot")
print("Type 'bye' to exit the conversation")
print("="*50)

# Main chatbot loop
while True:
    # Get user input
    user_input = input("\nYou: ").lower()

    # Exit condition
    if user_input == "bye":
        print("\nChatbot: Goodbye! Have a great day!")
        break

    # Initialize response variable
    response = "I'm sorry, I didn't understand that. Can you try asking something else?"

    # Check for matches in our response dictionary
    for pattern in responses:
        if pattern in user_input:
            response = responses[pattern]
            break

    # Display the response
    print(f"\nChatbot: {response}")
EOFcat > chatbot.py <<'EOF'
# Dictionary mapping user inputs to bot responses
responses = {
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I'm just a program, but I'm functioning as expected!",
    "what is your name": "I'm SimpleBot, a rule-based chatbot.",
    "what can you do": "I can answer simple questions based on predefined patterns.",
    "bye": "Goodbye! Have a great day!",
}

# Display welcome message
print("="*50)
print("Simple Rule-Based Chatbot")
print("Type 'bye' to exit the conversation")
print("="*50)

# Main chatbot loop
while True:
    # Get user input
    user_input = input("\nYou: ").lower()

    # Exit condition
    if user_input == "bye":
        print("\nChatbot: Goodbye! Have a great day!")
        break

    # Initialize response variable
    response = "I'm sorry, I didn't understand that. Can you try asking something else?"

    # Check for matches in our response dictionary
    for pattern in responses:
        if pattern in user_input:
            response = responses[pattern]
            break

    # Display the response
    print(f"\nChatbot: {response}")
