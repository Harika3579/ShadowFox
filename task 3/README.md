# 🧠 Task 3: AI-Driven Natural Language Processing (NLP)

## 📌 Project Overview
This project fulfills the Advanced Task requirement for the **ShadowFox AIML Internship**. The objective was to deploy a Language Model (LM), implement it from scratch using PyTorch and Hugging Face `transformers`, and conduct an in-depth analysis of its underlying architecture and attention mechanisms.

## 🛠️ Methodology & Implementation
* **Model Selection:** Selected `distilgpt2` for its efficiency in text generation while retaining complex architectural features comparable to larger GPT models.
* **Implementation:** Bypassed high-level APIs to manually load the `AutoModelForCausalLM` and `AutoTokenizer`, explicitly configuring the model to output its hidden attention states.
* **Token Analysis:** Translated raw text prompts into tensor representations to analyze how the model processes sequential data.

## 📊 Exploration and Visualizing Attention
Instead of treating the LM as a black box, this project focused on **Explainable AI (XAI)**. 
* Extracted the multidimensional attention matrices during a forward pass.
* Utilized `seaborn` and `matplotlib` to generate a heatmap of the final layer's attention weights. 
* **Findings:** The heatmap visually proves how the model maintains contextual understanding by assigning distinct mathematical weights to prior tokens (e.g., matching verbs to their preceding subjects).

## 💡 Conclusion and Broader Implications
* **Capabilities & Limitations:** While `distilgpt2` generates coherent short-form text and demonstrates strong syntactical attention, it is still constrained by its training data limits, occasionally losing deeper semantic context over longer generations.
* **Real-World Applications:** Understanding these attention mechanisms is critical for debugging AI hallucinations, improving prompt engineering, and fine-tuning models for specialized domains like healthcare or legal tech.
* **Ethical Considerations:** As models become more advanced, visualizing attention helps ensure that AI is making decisions based on relevant context rather than ingrained biases within the training data.

## 🔗 Proof of Work
* **Task screenshots: https://drive.google.com/drive/folders/1Bp12KCPhRNRQs2RrwVwee_e1iQIC8SGd?usp=sharing
