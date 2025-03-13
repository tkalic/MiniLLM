import json
from transformers import pipeline

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2", truncation=True)

# Load training data from JSON file
def load_training_data():
    with open("training_data.json", "r") as file:
        return {entry["input"]: entry["output"] for entry in json.load(file)}

custom_data = load_training_data()

def generate_text(prompt, max_length=50):
    """
    Generates text based on the given prompt.
    Uses predefined answers for known questions, otherwise GPT-2 generates text.

    Parameters:
    prompt (str): The input text that serves as the starting point.
    max_length (int): The maximum length of the generated text.

    Returns:
    str: The generated text.
    """

    if prompt in custom_data:
        return custom_data[prompt]
    
    result = generator(prompt, max_length=max_length, pad_token_id=50256)
    return result[0]["generated_text"]

if __name__ == "__main__":
    while True:
        user_input = input("\nAsk me about my GitHub project (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        output = generate_text(user_input)
        print("\nGenerated Response:\n", output)
