# MiniLLM
Programming a mini simplified version of a large language model

## 🔧 Fix: Handling Truncation and Padding Issues

### 📝 Problem: Warning messages about truncation and padding
When running the script, you might have encountered a warning like this:


This happens because:
1. The model needs explicit instructions to **truncate** (shorten) input texts that are too long.
2. GPT-2 **does not have a default padding token**, which can cause issues when processing text.

### ✅ Solution: Explicitly setting truncation and pad_token_id

To resolve this, we added two parameters in `main.py`:

```python
generator = pipeline("text-generation", model="gpt2", truncation=True)

I fed the model with some Q&A data in training_data.js

### Testing:

Getting the custom data from my user input prompt requires to be completely exact, even with upper and lower case letters.
If you write "What is your GitHub project about?" you get as an output: "My project is an AI chatbot using GPT-2."
But if you only mess up the upper cases in the word "GitHub" and write: "What is your github project about?" you get the answer: "Forgive me for being such an obnoxious guy but I feel like I've been getting a little too loud with the last few paragraphs and it's finally over…I want everyone to know that I'm" what is quite interesting

