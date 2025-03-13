# Import necessary libraries
from transformers import pipeline

# Load the GPT-2 model using Hugging Face's pipeline
# `truncation=True` ensures that long input texts are automatically shortened.
generator = pipeline("text-generation", model="gpt2", truncation=True)

def generate_text(prompt, max_length=50):
    """
    Generates text based on the given prompt.

    Parameters:
    prompt (str): The input text that serves as the starting point.
    max_length (int): The maximum length of the generated text.

    Returns:
    str: The generated text.
    """

    # Generate text using the model
    result = generator(
        prompt, 
        max_length=max_length, 
        pad_token_id=50256  # GPT-2 does not have a default pad token, so we use the EOS token (50256)
    )
    
    return result[0]["generated_text"]

if __name__ == "__main__":
    # Prompt input from the user
    user_input = input("Enter a prompt: ")

    # Generate and print the text
    output = generate_text(user_input)
    print("\nGenerated Text:\n", output)
