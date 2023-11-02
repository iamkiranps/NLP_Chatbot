"""
    Kiran Ponappan Sreekumari
    CSC525 – Principles of Machine Learning
    Colorado State University - Global
    Dr. Joseph Issa
    October 13, 2023
    Portfolio Milestone Project: NLP Chatbot
    **** Program to preprocess Cornell Movie Coverstation Corpus ****
"""
import argparse
import os
import math
from arguments import Args

def process_movie_lines(data_dir):
    """
    Process movie lines in key-value pairs
    """
    print(f'Processing {os.path.join(data_dir,"movie_lines.txt")} ...')
    movie_lines_kv = {}
    with open(os.path.join(data_dir,"movie_lines.txt"),"r", encoding="ISO-8859-1") as mfile:
        for line in mfile:
            line = line.strip().split(" +++$+++ ")
            lineid = line[0]
            dialog = line[-1]
            movie_lines_kv[lineid]=dialog
    return movie_lines_kv

def process_movie_converations(data_dir):
    """
    Process movie converastaions into list of dialog ids
    """
    print(f'Processing {os.path.join(data_dir,"movie_conversations.txt")} ...')
    conversations = []
    with open(os.path.join(data_dir, "movie_conversations.txt"), "r", encoding="ISO-8859-1") as conv_file:
        for line in conv_file:
            line = line.strip().split(" +++$+++ ")[-1]
            dialog_ids = eval(line)
            conversations.append(dialog_ids)
    return conversations

def convert_to_conversations(data_dir, movie_lines_kv, conversations):
    """
    Write the output into chatbot conversational format
    """
    output_file = "cornell_movie_dialogs_conversations.txt"
    print(f'Creating {os.path.join(data_dir,output_file)} ...')
    # Create conversational format
    with open(os.path.join(data_dir,output_file), "w", encoding="utf-8") as out_file:
        for conversation in conversations:
            for i in range(0, len(conversation), 2):
    #             print(i,conversation[i], "User : ", movie_lines_kv[conversation[i]])
                user_utterance = movie_lines_kv[conversation[i]]
                if i+1 < len(conversation):
                    bot_utterance = movie_lines_kv[conversation[i+1]]
    #                 print(i+1, conversation[i+1], "Bot : ", movie_lines_kv[conversation[i+1]])
                else:
                    None
                conv = f"User: {user_utterance}\nBot: {bot_utterance}\n"
                out_file.write(conv)
    print(f'Data processing completed')


def split_train_test(data_dir, input_file, split_ratio:float = 0.7):
    """Split the file into train and validation files"""
    print(f'Creating train file train.txt and validation file validation.txt ...')
    with open(os.path.join(data_dir, input_file), 'r', encoding='utf-8') as file,\
            open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf-8') as train, \
            open(os.path.join(data_dir, 'validation.txt'), 'w', encoding='utf-8') as validation:
        conversations = file.read().split('\n')
        split_index=math.floor(len(conversations)*split_ratio)
        train.writelines('\n'.join(conversations[:split_index]))
        validation.writelines('\n'.join(conversations[split_index:]))

if __name__ == '__main__':

    args = Args()
    # Path to the dataset directory
    data_dir = args.data_dir

    # Output file for preprocessed conversations
    output_file = args.corpus_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=data_dir, help="Path of the folder containing the Cornell Movie Dataset.")

    args = parser.parse_args()
    movie_lines = process_movie_lines(data_dir)
    conversations = process_movie_converations(data_dir)
    convert_to_conversations(data_dir, movie_lines, conversations)
    split_train_test(data_dir, output_file)