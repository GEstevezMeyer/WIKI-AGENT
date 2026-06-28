import json 
import os 
from multiprocessing import Pool
from datasets import IterableDataset,IterableDatasetDict,load_dataset
from tqdm import tqdm 

def _encode_huggingface_dataset_standalone(dataset: IterableDataset) -> list[int]:
    tokens: list[int] = []
    for row in tqdm(dataset, desc="encoding"):
        tokens.extend(row["text"].encode("utf-8"))
    return tokens



class Tokenizer:
    def __init__(self,padding_length:int = 512,not_found_token:int = 0):
        self._merges:dict[tuple[int,int],int] = {}
        self._vocabulary:dict[int,bytes] = {}
        self.padding_length: int = padding_length
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
    
    
    def _encode_hugginface_dataset_multiprocessing(self,dataset:IterableDataset):
        cpu_count:int = int(os.environ.get("cpu_count",1))
        n_process = cpu_count
        if cpu_count > dataset.num_shards:
            n_process = dataset.num_shards

        divided_dataset:list[IterableDataset] = [dataset.shard(num_shards=cpu_count,index=i) for i in range(n_process)]
        res_tokens:list[int] = []

        with Pool(n_process) as p:
            divided_tokens:list[list[int]] = p.map(_encode_huggingface_dataset_standalone,divided_dataset)

        for tokens in tqdm(divided_tokens,desc="merging tokens"):
            res_tokens.extend(tokens)

        return res_tokens

    @staticmethod
    def _encode_huggingface_dataset(dataset:IterableDataset) -> list[int]:
        tokens:list[int] = [] 
        for row in tqdm(dataset,desc="encoding hugging face dataset"):
            tokens.extend(row["text"].encode("utf-8"))
        return tokens

    @staticmethod
    def _encode_raw_dataset(dataset:list[str]) -> list[int]:
        tokens:list[int] = []
        for text in tqdm(dataset,desc= "Encoding dataset into utf-8: "):
            tokens.extend(text.encode("utf-8"))

        return tokens

    def fit(self,dataset:list[str] | IterableDataset,n_merges:int = 1000,multiproces:bool = False) -> None:
        if type(dataset) == list:
            tokens:list[int] = self._encode_raw_dataset(dataset)
        elif dataset.num_shards == 1 or not multiproces:
            tokens:list[int] = self._encode_huggingface_dataset(dataset)
        else:
            tokens:list[int] = self._encode_hugginface_dataset_multiprocessing(dataset)

        for _ in tqdm(range(n_merges),desc= "fitting tokenizer"):
            counts:dict[tuple[int, int], int] = self._get_stats(tokens)
            mcp:int = max(counts,key = counts.get)
            self._last_token+=1
            self._merges[mcp] = self._last_token
            self._vocabulary[self._last_token] = self._vocabulary[mcp[0]] + self._vocabulary[mcp[1]]

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
    dataset:IterableDataset = load_dataset("wikimedia/wikipedia", "20231101.en",streaming=True)
    train_tokenizer("wiki_agentic/tokenizer.json",dataset)

if __name__ == "__main__":
    main()