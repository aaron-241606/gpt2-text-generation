"""
Task-01: Generate text using fine-tuned GPT-2
Prodigy Infotech Internship
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_text(
    model_path,
    prompt,
    max_length=200,
    num_sequences=1,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
):
    """Generate text from a fine-tuned GPT-2 model."""

    print(f"📦 Loading model from: {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🖥️  Using device: {device}")

    print(f"\n📝 Prompt: \"{prompt}\"\n")
    print("=" * 60)

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    for i, output in enumerate(outputs):
        generated = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n🤖 Generated Text #{i + 1}:\n")
        print(generated)
        print("\n" + "=" * 60)

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using fine-tuned GPT-2")
    parser.add_argument("--model", type=str, default="./gpt2-finetuned", help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--num_sequences", type=int, default=1, help="Number of text sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (higher = more creative)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")

    args = parser.parse_args()

    generate_text(
        model_path=args.model,
        prompt=args.prompt,
        max_length=args.max_length,
        num_sequences=args.num_sequences,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
