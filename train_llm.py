from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

# Lade OpenWebText, aber nur die ersten 10.000 Einträge
dataset = load_dataset("openwebtext", trust_remote_code=True)

# Reduziere die Datenmenge
small_dataset = dataset["train"].select(range(10000))  # Nur 10.000 

# GPT-2 Modell & Tokenizer laden
model_name = "gpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fix: Setze Padding-Token auf das EOS-Token
tokenizer.pad_token = tokenizer.eos_token

# Tokenisierung der Trainingsdaten
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()  # Labels = Input-Text für Training
    return tokens


tokenized_dataset = small_dataset.map(tokenize_function, batched=True)

# Training-Parameter
training_args = TrainingArguments(
    output_dir="./trained_model",
    evaluation_strategy="no", # Deaktiviert die Evaluierung
    per_device_train_batch_size=2,  # Weniger GPU-RAM-Verbrauch
    num_train_epochs=2,  # Weniger Training für Schnelligkeit
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Speichern des trainierten Modells
model.save_pretrained("./my_mini_llm")
tokenizer.save_pretrained("./my_mini_llm")
