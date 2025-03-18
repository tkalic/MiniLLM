from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Lade dein trainiertes Modell
model_path = "./my_mini_llm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Erstelle eine Pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    user_input = input("\nEnter a prompt (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    output = generator(user_input, max_length=50, truncation=True)
    print("\nGenerated Response:\n", output[0]["generated_text"])
