import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing import Preprocessing

class SquadDataset(Dataset):
    """Dataset object for the Stanford Question Answering Dataset. 
    
    
    """
    
    def __init__(self, data_file_path, glove_file_path, target):
        """
        Args:
            json_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = pd.read_json(data_file_path, orient='records')['data']
        self.dataset = self.flatten_data(self.dataset)
        self.preprocess = Preprocessing(glove_file_path)
        self.target = target

        
    def __len__(self):
        return len(self.dataset)

    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if self.target in ['text','question']:
            data_point = item[self.target]
            data_point = self.preprocess.preprocess(data_point)
            sample = {self.target: torch.from_numpy(data_point)}
        else:
            data_point = item['answer']
            sample = {self.target: data_point}
        
        return sample
    
    
    def flatten_data(self, data):
        flat_data = []
        for article in data:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    flat_data.append({'text': paragraph['context'], 
                                      'question': qa['question'], 
                                      'answer': qa['answers'][0]['text']})
        return flat_data



