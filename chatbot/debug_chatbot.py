# A simple script to enable debug mode in our chatbot
import os

# Set environment variable to enable debug mode
os.environ['CHATBOT_DEBUG'] = 'True'

# Print instructions
print("Debug mode enabled for chatbot.")
print("Run your chatbot script (python3 ml_chatbot.py) to see intent predictions and confidence scores.")
print("This will help you evaluate and improve your chatbot's performance.")
