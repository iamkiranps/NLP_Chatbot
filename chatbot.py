"""
    Kiran Ponappan Sreekumari
    CSC525 – Principles of Machine Learning
    Colorado State University - Global
    Dr. Joseph Issa
    October 13, 2023
    Portfolio Milestone Project: NLP Chatbot
    **** Program to invoke the chatbot application ****
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from arguments import Args

def generate_response(input_text, conversation_history=[]):
    """
    Generates ChatBot Responses
    """
    args = Args()
    # Encode the conversation history and user input
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors=args.tensors, padding=args.padding)

    # Extend the conversation history with the user input
    conversation_history.append(input_ids)

    # Generate a response
    with torch.no_grad():
        response_ids = model.generate(
            input_ids=input_ids,
            max_length=args.generate_max_length,  # Adjust the maximum length as needed
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=args.do_sample, # With sampling, the output becomes more creative!,
            # max_new_tokens=50, # Setting `max_new_tokens` allows you to control the maximum length
            num_beams=args.num_beams,
            early_stopping=args.early_stopping, #Generation is finished when all beam hypotheses reached the EOS token.
            no_repeat_ngram_size=args.no_repeat_ngram_size, #The most common n-grams penalty makes sure that no n-gram appears twice by manually setting the probability of next words that could create an already seen n-gram to 0.
            top_k=args.top_k, 
            top_p=args.top_p,
            temperature = args.temperature
        )

    # Decode the response
    response = tokenizer.decode(response_ids[0], skip_special_tokens=args.skip_special_tokens, padding_side=args.padding_side)

    return response, conversation_history

if __name__ == '__main__':
    args = Args()

    # Load the model and tokenizer from disk
    model = AutoModelForCausalLM.from_pretrained(args.save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.save_path)
    padding_side=args.padding_side

    # Initialize the conversation history
    conversation_history = []

    # Chatbot interaction loop
    print("Chatbot: Hello! I'm your chatbot. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        response, conversation_history = generate_response(user_input, conversation_history)
        response = response[len(user_input):]
        # print("You:", user_input)
        print("Chatbot:", response)
