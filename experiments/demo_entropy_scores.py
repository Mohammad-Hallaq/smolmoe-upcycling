# experiments/demo_entropy_scores.py

from transformers import AutoTokenizer

from moe_llm.data_selection import text_per_token_entropy


def main():
    text1 = "Hi my name is Mohammad."
    text2 = "my name is " * 30
    text3 = "Hi my name is " * 5

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    s1 = text_per_token_entropy(text1, tokenizer)
    s2 = text_per_token_entropy(text2, tokenizer)
    s3 = text_per_token_entropy(text3, tokenizer)

    print("Entropy scores (bits/token):")
    print(f"text1: {text1} \nhas a score of {s1:.3f}\n")
    print(f"text2: {text2} \nhas a score of {s2:.3f}\n")
    print(f"text3: {text3} \nhas a score of {s3:.3f}")


if __name__ == "__main__":
    main()
