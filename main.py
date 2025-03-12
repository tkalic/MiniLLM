# Import necessary libraries
from transformers import pipeline

# Load the GPT-2 model using Hugging Face's pipeline
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, max_length=50):
    """
    Generates text based on the given prompt.
    
    Parameters:
    prompt (str): The input text that serves as the starting point.
    max_length (int): The maximum length of the generated text.
    
    Returns:
    str: The generated text.
    """
    result = generator(prompt, max_length=max_length)
    return result[0]["generated_text"]

if __name__ == "__main__":
    # Example prompt
    user_input = input("Enter a prompt: ")
    output = generate_text(user_input)
    print("\nGenerated Text:\n", output)

