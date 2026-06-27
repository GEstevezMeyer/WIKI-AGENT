import re
import json 
import os 

from datasets import IterableDataset
from wiki_agentic.utils import divide_dataset_shards
from multiprocessing import Process,Queue
from functools import partial
from tqdm import tqdm 


class Tokenizer:
    def __init__(self,padding_lenght:int = 512,not_found_token:int = 0):
        self._merges:dict[tuple[int,int],int] = {}
        self._vocabulary:dict[int,bytes] = {}
        self.padding_lenght:int = padding_lenght
        self.not_found_token:int = not_found_token
        self._last_token = 256

        for i in range(256):
            self._vocabulary[i] = bytes([i])

    @staticmethod
    def _get_stats(tokens: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    
        new_ids: list[int] = []
        i: int = 0
        while i < len(ids):
            
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2  
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _encode_raw_dataset(dataset:list[str]) -> list[int]:
        tokens:list[int] = []
        for text in dataset:
            tokens.extend(text.encode("utf-8"))

        return tokens

    def fit(self,dataset:list[str],n_merges:int = 1000) -> None:
        tokens:list[int] = self._encode_raw_dataset(dataset)

        for _ in tqdm(range(n_merges),desc= "fitting tokenizer"):
            counts:dict[tuple[int, int], int] = self._get_stats(tokens)
            mcp:int = max(counts,keys = counts.get)
            self._last_token+=1
            self.merge[mcp] = self._last_token
            self._vocabulary = (self._vocabulary[mcp[0]],self.self._vocabulary[mcp[1]])

            tokens:list[int] = self._merge_pair(tokens,mcp,self._last_token)


    def encode(self, text: str) -> list[int]:
        tokens: list[int] = list(text.encode("utf-8"))

        for pair, new_id in self._merges.items():
            tokens = self._merge_pair(tokens, pair, new_id)

        return tokens
    
    def decode(self, ids: list[int]) -> str:
        
        byte_sequence = b"".join(
            self._vocabulary[token_id]
            for token_id in ids
            if token_id in self._vocabulary
        )
        return byte_sequence.decode("utf-8", errors="replace")
    
    def _padding(self, token_list: list[int]) -> list[int]:
        if self.padding_length < len(token_list):
            return token_list[:self.padding_length]       
        padding = [0] * (self.padding_length - len(token_list))

        return token_list + padding                       

    def transform(self, dataset: list[str]) -> list[list[int]]:
        return [self._padding(self.encode(text)) for text in dataset]
    
    def save_tokenizer(self, path: str) -> None:
        data = {
            "merges": {f"{a},{b}": new_id for (a, b), new_id in self._merges.items()},
            "padding_length": self.padding_length,
            "not_found_token": self.not_found_token,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    

    def load_tokenizer(self, path: str) -> None:
        
        with open(path, "r") as f:
            data = json.load(f)

        self.padding_length = data["padding_length"]
        self.not_found_token = data["not_found_token"]

        
        self._merges = {}
        self._vocabulary = {i: bytes([i]) for i in range(256)}
        self._last_token = 255

        for key, new_id in data["merges"].items():
            a, b = map(int, key.split(","))
            pair = (a, b)
            self._merges[pair] = new_id
            self._vocabulary[new_id] = (
                self._vocabulary[a] + self._vocabulary[b]
            )
            self._last_token = max(self._last_token, new_id)
    



def merge_vocabularies(base: Tokenizer, vocabularies: list[dict]) -> None:
    for vocab in tqdm(vocabularies, desc="Merging vocab"):
        for word, _ in vocab.items():
            if word not in base._found_word:
                base._last_token += 1
                base._vocabulary[word] = base._last_token
                base._found_word.add(word)

