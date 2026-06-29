import json 
import os 
from multiprocessing import Pool
from datasets import IterableDataset,IterableDatasetDict,load_dataset
from tqdm import tqdm 



class Tokenizer:
    def __init__(self,padding_length:int = 512,not_found_token:int = 0):
        self.counts = {}
        self._merges:dict[tuple[int,int],int] = {}
        self._vocabulary:dict[int,bytes] = {}
        self.padding_length: int = padding_length
        self.not_found_token:int = not_found_token
        self._last_token = 256

        for i in range(256):
            self._vocabulary[i] = bytes([i])

    def _get_stats_from_word_freqs(self) -> None:
        self.counts = {}
        for word_tokens, freq in self.word_freqs.items():
            for pair in zip(word_tokens, word_tokens[1:]):
                self.counts[pair] = self.counts.get(pair, 0) + freq

    def _build_word_freqs(self, dataset: IterableDataset) -> None:
        self.word_freqs = {}
        for row in tqdm(dataset, desc="building word frequencies"):
            for word in row["text"].split():
                token_tuple = tuple(word.encode("utf-8"))
                self.word_freqs[token_tuple] = self.word_freqs.get(token_tuple, 0) + 1

    def _build_word_freqs_raw(self, dataset: list[str]) -> None:
        self.word_freqs = {}
        for text in tqdm(dataset, desc="building word frequencies"):
            for word in text.split():
                token_tuple = tuple(word.encode("utf-8"))
                self.word_freqs[token_tuple] = self.word_freqs.get(token_tuple, 0) + 1

    def _merge_word_freqs(self, pair: tuple[int, int]) -> None:
        new_word_freqs = {}
        for word_tokens, freq in self.word_freqs.items():
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == pair[0] and word_tokens[i+1] == pair[1]:
                    new_tokens.append(self._last_token)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_word_freqs[tuple(new_tokens)] = freq
        self.word_freqs = new_word_freqs

    def fit(self, dataset: list[str] | IterableDataset, n_merges: int = 1000) -> None:
        if isinstance(dataset, list):
            self._build_word_freqs_raw(dataset)
        else:
            self._build_word_freqs(dataset)

        self._get_stats_from_word_freqs()

        for _ in tqdm(range(n_merges), desc="fitting tokenizer"):
            if not self.counts:
                break
            mcp = max(self.counts, key=self.counts.get)
            self._last_token += 1
            self._merges[mcp] = self._last_token
            self._vocabulary[self._last_token] = self._vocabulary[mcp[0]] + self._vocabulary[mcp[1]]
            self._merge_word_freqs(mcp)
            self._get_stats_from_word_freqs()

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
        padding_length = [0] * (self.padding_length - len(token_list))

        return token_list + padding_length                       

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
    
    def __len__(self):
        return len(list(self._vocabulary.keys()))

def train_tokenizer(path:str,dataset:IterableDataset | list[str],padding_length = 512, 
                    not_found_token:int = 0,n_merges:int = 1000):
    tokenizer:Tokenizer = Tokenizer(padding_length=padding_length,not_found_token=not_found_token)

    if isinstance(dataset,IterableDatasetDict):
        dataset = dataset["train"]

    tokenizer.fit(dataset,n_merges=n_merges)
    tokenizer.save_tokenizer(path)

    print(f"Tokenizer trained the len of the vocab is {len(tokenizer)}")
    

def main():
    dataset:IterableDataset = load_dataset("wikimedia/wikipedia", "20231101.en",streaming=True)["train"]
    dataset = dataset.take(100_000)
    train_tokenizer("src/tokenizer.json",dataset)

