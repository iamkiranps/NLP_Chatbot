"""
    Kiran Ponappan Sreekumari
    CSC525 – Principles of Machine Learning
    Colorado State University - Global
    Dr. Joseph Issa
    October 13, 2023
    Portfolio Milestone Project: NLP Chatbot
    **** Program to build model with DialogGPT pretrained model and Cornell Movie Coverstation Corpus ****
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import os
from arguments import Args
from transformers.utils import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
set_seed(42)

# logging.set_verbosity_info()
# logger = logging.get_logger("transformers")
# logger.info("INFO")
# logger.warning("WARN")

def build_model(model_name, file_path):
    """
    Build the model
    """
    args = Args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    padding_side = args.padding_side

    # Load the DialoGPT model and tokenizer
    # model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side=padding_side, padding=args.padding, max_length=args.max_length)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Set the padding token to the left side
    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess conversational dataset
    # Replace with dataset's file path
    with open(os.path.join(file_path, args.corpus_file), 'r', encoding='utf-8') as file:
        conversations = file.read().split('\n')
    
    # Add special token to tokenizer
    tokenizer.add_tokens(["Bot:","User:"])

    # Tokenize and preprocess the data
    # tokenized_conversations = [tokenizer.encode(conv,max_length=args.max_length, return_tensors=args.tensors, add_special_tokens=args.add_special_tokens, truncation=args.truncation) for conv in conversations[0:10]]
    
    tokenized_conversations = [tokenizer.encode(conv, max_length=40, return_tensors="pt", padding="max_length",add_special_tokens=True, truncation=True) for conv in conversations[:10]]
    # Create a DataLoader for dataset
    dataset = torch.cat(tokenized_conversations, dim=0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # Define optimizer and loss function (e.g., CrossEntropyLoss)

    # Training loop
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    total_loss = 0.0
    perplexity = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    progress_bar = tqdm(range(num_epochs))
    
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch.to(device)

            # Forward pass
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

        # Calculate perplexity [perplexity measures how likely the model is to generate the input text sequence]
        perplexity = total_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs}: Loss - {loss:.4f}   Perplexity - {perplexity:.4f}")

    # Save the fine-tuned model
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(model.config)

if __name__ == '__main__':
    """
    Initialize arguments and run the model
    """
    args = Args()
    model_name = args.model_name

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_type", type=str, default='medium', help="Select DialogGPT model 'small','medium','large'")
    # parser.add_argument("--dataset_path", type=str, default='./Cornell_Movie_Dialogs_Corpus/', help="Path of the source file")

    # args=parser.parse_args()
    # model_name = f"microsoft/DialoGPT-{args.model_type}"

    build_model(model_name=model_name, file_path=args.data_dir)
