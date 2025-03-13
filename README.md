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

