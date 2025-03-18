from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# Stelle sicher, dass der Pfad zum lokalen Modell korrekt ist
model_path = os.path.abspath("./my_mini_llm")

# Lade Tokenizer & Modell explizit von lokalem Pfad
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# Erstelle eine Pipeline für die Generierung
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Teste das Modell mit einer Eingabe
prompt = "The future of AI is"
result = generator(prompt, max_length=50)

print("\nGenerated Response:\n", result[0]["generated_text"])
