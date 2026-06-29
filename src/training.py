import torch
from torch.utils.data import DataLoader
from src.pipeline import main_pipeline
from src.architecture import Transformer
from src.utils import load_config



def training(tokenizer_path:str = "src/tokenizer.json",config_path:str = "src/config.toml"):
    config:dict = load_config(config_path)["model"]

    pipeline:tuple[DataLoader,int,int] = main_pipeline(tokenizer_path,config_path)
    dataloader,vocab_size,seq_len = pipeline

    model = Transformer(src_vocab_size=vocab_size,tgt_vocab_size=vocab_size,seq_len=seq_len,**config)

    

if __name__ == "__main__": 
    training()