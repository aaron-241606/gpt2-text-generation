# Task-01: Text Generation with GPT-2

> Train a model to generate coherent and contextually relevant text based on a given prompt.

---

## 📌 Overview

This project fine-tunes **GPT-2**, a transformer model developed by OpenAI, on a custom dataset to generate text that mimics the style and structure of the training data.

You will learn how to:
- Load and preprocess a custom text dataset
- Fine-tune GPT-2 using HuggingFace Transformers
- Generate contextually relevant text from a prompt

---

## 🗂️ Project Structure

```
gpt2-text-generation/
├── data/
│   └── custom_dataset.txt      # Your training text data
├── train.py                    # Fine-tuning script
├── generate.py                 # Text generation script
├── requirements.txt            # Dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/aaron-241606/gpt2-text-generation.git
cd gpt2-text-generation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place your training text in `data/custom_dataset.txt`. Each line can be a sentence or paragraph.

---

## 🚀 Usage

### Fine-tune GPT-2
```bash
python train.py --dataset data/custom_dataset.txt --epochs 3 --output ./gpt2-finetuned
```

### Generate Text
```bash
python generate.py --model ./gpt2-finetuned --prompt "Once upon a time" --max_length 200
```

---

## 🧠 Model Details

| Property       | Value              |
|----------------|--------------------|
| Base Model     | GPT-2 (OpenAI)     |
| Framework      | HuggingFace Transformers |
| Task           | Causal Language Modeling |
| Fine-tuning    | Custom Dataset     |

---

## 📚 References

- [#1 GPT-2 Paper – Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-unsupervised)
- [#2 HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)

---

## 🏢 Credits

**Prodigy Infotech** – Task-01 Internship Project
