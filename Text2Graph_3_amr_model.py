from transformers import BartTokenizer, BartModel
import torch

# === Replace this with your local model directory ===
model_dir = "Models/facebook_bart_large"

print("Loading tokenizer and model from local path...")
tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartModel.from_pretrained(model_dir)

# === Try a sample input ===
sentence = "Dr. Smith sent a PET scan order."

inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print("\n✅ Model loaded and executed successfully!")
print("Last hidden state shape:", outputs.last_hidden_state.shape)
