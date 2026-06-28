import torch
from torch.utils.data import IterableDataset as IterableTorchDataset
from datasets import IterableDataset as IterableHuggingDataset
from wiki_agentic.tokenizer import Tokenizer


class WikipediaDataset(IterableTorchDataset):
    def __init__(self,tokenizer:Tokenizer,dataset:IterableHuggingDataset):
        super().__init__()
        self.tokenizer:Tokenizer = tokenizer
        self.dataset:IterableHuggingDataset = dataset
        
    def __iter__(self):
            for row in self.dataset:
                tokens = self.tokenizer._padding(self.tokenizer.encode(row["text"]))
                yield torch.tensor(tokens, dtype=torch.long)