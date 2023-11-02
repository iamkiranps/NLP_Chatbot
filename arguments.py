class Args():
    def __init__(self) -> None:
        """
        Define all parameters
        """
        ##Environment parameters
        self.data_dir = "./Cornell_Movie_Dialogs_Corpus"
        self.corpus_file = "cornell_movie_dialogs_conversations.txt"
        

        ## Model parameters
        self.model_name = "microsoft/DialoGPT-medium"
        self.save_path = "./fine_tuned_dialogpt"

        ##Training parameters
        self.num_epochs = 10
        self.learning_rate = 2e-5

        ##Tokenizer parameters
        self.padding_side="left"
        self.padding=True
        self.max_length=40
        self.tensors="pt"
        self.add_special_tokens=True
        self.truncation=True

        ##Dataloader parameters
        self.batch_size=4
        self.shuffle=True
        self.num_workers=2

        ##Chat Generation parameters
        self.generate_max_length=100
        self.skip_special_tokens=True
        self.do_sample=True
        self.num_beams=2
        self.early_stopping=True
        self.no_repeat_ngram_size=3
        self.top_k=100
        self.top_p=0.7
        self.temperature = 0.8        
