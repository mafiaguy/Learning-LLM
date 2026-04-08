from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
revision_id = "6e505f907968c4a9360773ff57885cdc6dca4bfd"
model = AutoModelForSeq2SeqLM.from_pretrained(
        "Falconsai/text_summarization",  # Using a model suitable for summarization
        revision=revision_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        )
tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization", revision=revision_id)
from transformers import pipeline

summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=150,  # Adjust max_length as needed for summarization
        min_length=30,
        do_sample=False
        )
user_input = input("Enter the text you want to summarize:")
response = summarizer(user_input)
print(response[0]["summary_text"])
while True:
    print("-" *50) # Horizontal line
    user_input = input("\033[92mEnter the text you want to summarize: \033[0m") # Ask the user for input in green color
    if user_input in ['X', 'x']: # If user types X or x, exit the program
        print("Exiting.")
        break
    else:
        print("-" *50)
        print("Calling LLM for summarizing")
        response = summarizer(user_input)
        print(response[0]["summary_text"])
