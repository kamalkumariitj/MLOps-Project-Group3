#!/usr/bin/env python3
"""Small smoke-test runner: loads a tiny ANLI sample and tokenizes it with RoBERTa.

Usage: python run_smoke.py
"""
from datasets import load_dataset
from transformers import RobertaTokenizer

def main():
    print("Loading a small ANLI sample (50 examples)...")
    ds = load_dataset("facebook/anli")
    # take 50 examples from train_r1
    split = ds['train_r1'].select(range(min(50, len(ds['train_r1']))))
    texts = [f"premise: {ex['premise']} hypothesis: {ex['hypothesis']}" for ex in split]

    print("Loading tokenizer (roberta-base)...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    print(f"Tokenizing {len(texts)} examples (max_length=128)")
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)

    print('Sample tokenized input_ids length:', len(enc['input_ids'][0]))
    print('Done.')

if __name__ == '__main__':
    main()
