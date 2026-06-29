import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as IterableTorchDataset
from datasets import IterableDataset as IterableHuggingDataset
from datasets import load_dataset
from src.tokenizer import Tokenizer
from src.utils import load_config


class WikipediaDataset(IterableTorchDataset):
    def __init__(self,tokenizer:Tokenizer,dataset:IterableHuggingDataset):
        super().__init__()
        self.tokenizer:Tokenizer = tokenizer
        self.dataset:IterableHuggingDataset = dataset
        
    def __iter__(self):
            for row in self.dataset:
                tokens = self.tokenizer._padding(self.tokenizer.encode(row["text"]))
                t = torch.tensor(tokens, dtype=torch.long)
                yield t[:-1], t[1:]


def create_WikipediaDataset(tokenizer_path:str) -> tuple[WikipediaDataset,int,int]:
    print("Loading Wikipedia in streaming:")
    dataset:IterableHuggingDataset = load_dataset("wikimedia/wikipedia", "20231101.en",streaming=True)["train"]
    print("Sucess loading Wikipedia in streaming")
    tokenizer:Tokenizer = Tokenizer()
    print("Loading Tokenizer: ")
    tokenizer.load_tokenizer(tokenizer_path)
    print("Sucess loading Tokenizer")
    wikidataset:WikipediaDataset = WikipediaDataset(tokenizer=tokenizer,dataset=dataset)
    vocab_size:int = len(tokenizer)
    seq_len:int = tokenizer.padding_length

    return wikidataset,vocab_size,seq_len


def create_DataLoader(dataset:IterableTorchDataset,config:dict[dict]) -> DataLoader:
    return DataLoader(dataset,**config)

def main_pipeline(tokenizer_path:str = "src/tokenizer.json",
                  config_path:str = "src/config.toml") -> tuple[DataLoader,int,int] :
    config:dict = load_config(config_path)["pipeline"]
    dataset,vocab_size,seq_len = create_WikipediaDataset(tokenizer_path)
    dataloader:DataLoader = create_DataLoader(dataset,config)
    return dataloader,vocab_size,seq_len
    



if __name__ == "__main__":
    test = main_pipeline()
    print(test)

    print(next(iter(test)))
