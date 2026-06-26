# ==========================================
# 1. IMPORT ADVANCED LIBRARIES
# ==========================================
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 2. LOAD MODEL & EXTRACT ARCHITECTURE
# ==========================================
print("Loading Model and Tokenizer...")
model_name = "distilgpt2"

# We load the tokenizer and the model manually. 
# CRITICAL: We set output_attentions=True to peek inside the neural network's brain.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
print("Advanced Architecture Loaded!")

# ==========================================
# 3. TEXT GENERATION & TOKEN ANALYSIS
# ==========================================
print("\n--- Tokenization & Generation ---")
prompt = "Artificial intelligence in the future will"

# Convert the text prompt into mathematical token IDs
inputs = tokenizer(prompt, return_tensors="pt")
print(f"Input Tokens: {tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}")

# Generate text using the model
outputs = model.generate(
    inputs.input_ids, 
    max_new_tokens=20, 
    do_sample=True, 
    temperature=0.7, 
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nFinal Generated Text:\n{generated_text}")

# ==========================================
# 4. VISUALIZING ATTENTION MECHANISMS
# ==========================================
print("\n--- Extracting Attention Weights ---")
# To visualize attention, we run a single forward pass of our prompt
with torch.no_grad():
    forward_pass = model(**inputs)

# Extract the attention matrices (Tuple of all layers)
attention_layers = forward_pass.attentions

# We will look at the last layer of the neural network, and its first "Attention Head"
last_layer_attention = attention_layers[-1] 
head_attention = last_layer_attention[0, 0].numpy() # Shape: (Seq_Len, Seq_Len)

# Get the exact tokens for our graph labels
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

# Plotting the Attention Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(head_attention, xticklabels=tokens, yticklabels=tokens, cmap='magma', annot=True, fmt=".2f")
plt.title("LM Attention Mechanism: What is the AI focusing on?")
plt.xlabel("Key Tokens (Being attended to)")
plt.ylabel("Query Tokens (Attending)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
