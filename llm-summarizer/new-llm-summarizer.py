import requests

while True:
    print("-" *50) # Horizontal line
    user_input = input("\033[92mEnter a file path (file://) or a URL (http:// or https://): \033[0m")

    if user_input in ['X', 'x']: # If user types X or x, exit the program
        print("Exiting.")
        break
    else:
        if user_input.startswith("file://"):
            with open(user_input[7:], 'r') as file:  # Skip 'file://' prefix
                user_input = file.read()
        elif user_input.startswith("http://") or user_input.startswith("https://"):
            response = requests.get(user_input)
            user_input = response.text
        else:
            print("Invalid input. Please start with file:// or http:///https://.")
            continue

        print("-" *50)
        print("Calling LLM")
        response = summarizer(user_input)
        print(response[0]["summary_text"])

