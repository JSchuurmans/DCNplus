import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from pycorenlp import StanfordCoreNLP


class Preprocessing():
    """
    Class containing tokenization and embeddings functions, borrowing from the Stanford
    CoreNLP tokenizer and the pretrained GloVe word embeddings
    
    About the CoreNLP server: if server is offline, run the command below in 
    Terminal from the stanford CoreNLP folder:
    
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000
    """
    
    
    def __init__(self, glove_file_path):
        self.annotator = StanfordCoreNLP('http://localhost:9001')
        self.embeddings = self.load_glove_embeddings(glove_file_path)
        
        
    def load_glove_embeddings(self, file_path):
        """
        Loads the glove word vectors from a textfile and parses it into a dictionary with 
        words and vectors.

        Returns:
        A dictionary of words and corresponding vectors
        """

        cached_file = '../data/glove.pickle'
        if os.path.isfile(cached_file): 
            print("Found pickled GloVe file. Loading...")
            with open(cached_file, 'rb') as handle:
                embeddings_dict = pickle.load(handle)
        else:
            print("Loading Glove Model from .txt...")
            with open(file_path,'r', encoding="utf8") as f:
                embeddings_dict = {}
                cnt = 0
                for i, line in enumerate(f):

                    split_line = line.split()

                    # Skip aberrant lines
                    if not len(split_line) == 301:
                        continue 

                    word = split_line[0]
                    embedding = np.array([float(val) for val in split_line[1:]])
                    embeddings_dict[word] = embedding
                    
            with open('../data/glove.pickle', 'wb') as handle:
                print("Saving GloVes as pickle...")
                pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done. {} words loaded!".format(len(embeddings_dict)))
        
        
        
        return embeddings_dict
    
    
    def preprocess(self, text):
        """
        Tokenizes and applies word embeddings to a text. Also pads or cuts the whole
        sequence to length 600.
        """
        tokenized_text = self.tokenize(text)
        embedded_text = self.embed(tokenized_text)
        embedded_text = embedded_text[:600,:]
        padded_embeddings = np.zeros([600, 300])
        padded_embeddings[:embedded_text.shape[0], :embedded_text.shape[1]] = embedded_text

        return padded_embeddings
    
    
    def tokenize(self, text):
        """
        Calls the Stanford CoreNLP Tokenizer running on a local server, which tokenizes 
        the input text.

        Returns:
        Tokenized text
        """
        annotated_text = self.annotator.annotate(text, 
                                                 properties={'annotators': 'tokenize', 
                                                              "outputFormat": "json"})
        tokenized_text = []
        for token in annotated_text['tokens']:
            word = token['word']
            tokenized_text.append(word)

        return tokenized_text
    
    
    def embed(self, words):
        """
        Takes words and returns corresponding GloVe word embeddings. Returns a zero vector 
        if no embedding is found.

        Returns:
        List of word vectors
        """
        word_vectors = np.zeros((len(words), 300))

        for i, word in enumerate(words):
            # Match word with vector
            try:
                vector = self.embeddings[word]
            except KeyError:
                # Set to zero vector if no match
                vector = np.zeros(300)

            word_vectors[i] = vector

        return word_vectors

    
    