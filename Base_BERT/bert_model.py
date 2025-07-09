from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model from Bio_ClinicalBERT and then load fine-tuned weights
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=6)

# Load the fine-tuned weights with strict=False to allow shape mismatches
state_dict = torch.load("./pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)

# Define emotion labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Input text
# List of input texts to test
texts = [
    "I feel very sad and alone.",
    "I'm so happy and grateful today!",
    "You make me feel loved and appreciated.",
    "This makes me so angry!",
    "I'm really scared of what might happen.",
    "Wow, what a surprise!",
    "I miss my family and it hurts.",
    "That dog is adorable and makes me smile.",
    "I can't believe this happened again!"
]

# Predict emotion for each input
model.eval()
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
        print(f"Text: {text}\n→ Predicted emotion: {labels[predicted_class]}\n")
