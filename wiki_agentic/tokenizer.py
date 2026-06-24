import re
import json 
import os 

from datasets import IterableDataset
from wiki_agentic.utils import divide_dataset_shards
from multiprocessing import Process,Queue
from functools import partial
from tqdm import tqdm 


class Tokenizer:
    def __init__(self,padding_lenght:int = 512,not_found_token = -1):
        self._vocabulary:dict = {}
        self._found_word:set = set()
        self._last_token:int = 0
        self.padding_lenght = padding_lenght
        self.not_found_token = not_found_token
        

    @property
    def vocabulary(self):
        return self._vocabulary
    
    @vocabulary.setter
    def vocabulary(self,new_vocab:dict[int]): 
        self._vocabulary = dict(sorted(new_vocab.items()))
        self._found_word = set(new_vocab.keys())
        self._last_token = self.vocabulary[list(self.vocabulary.keys())[-1]]

    def _cleaning_text(self,text:str) -> list[str]: 
        text:str = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]', '', text).lower()
        word_list:list[str] = text.split(" ")
        return  word_list

    def _fit_text(self,text:str) -> None:
        word_list:list[str] = self._cleaning_text(text)
        for word in word_list:
            if (word not in self._found_word) and (len(word) >= 1):
                self._last_token+= 1
                self._vocabulary[word] = self._last_token
                self._found_word.add(word)

    def fit(self,dataset:list[str]) -> None:
        for i in tqdm(range(len(dataset)),f"Fiting text"):
            self._fit_text(dataset[i])

    def fit_hugging_face(self,dataset:IterableDataset,shard_name = "0") -> None:
        position = int(shard_name)
        for row in tqdm(dataset,position=position,desc= f"Fitting shard_{shard_name}:"):
            self._fit_text(row["text"])

    def _padding(self,token_list:list[int]) -> list[int]:
        res:list[int] = []
        n:int = len(token_list)

        if self.padding_lenght < n:
            res.extend(token_list[:self.padding_lenght])
        else:
            res.extend(token_list)
            res.extend([0 for _ in range(self.padding_lenght-n)])

        return res 

    def _transform_text(self,text:str) -> list[int]:
        res:list[int] = []
        word_list:list[str] = self._cleaning_text(text)
        for word in word_list:
            if word in self._found_word:
                res.append(self.vocabulary[word])
            elif len(word) >= 1:
                res.append(self.not_found_token)

        return self._padding(res)
    
    def transform(self,dataset:list[str]) -> list[list[int]]:
        res:list[list[int]] = []
        for i in range(len(dataset)):
            res.append(self._transform_text(dataset[i]))
        return res 
    
    def save_tokenizer(self,path:str) -> None:
        with open(path, "w") as f:
            json.dump(self.vocabulary,f)
        
        
    def upload_tokenizer(self,path:str) -> None:
        with open(path,"r") as f:
            self.vocabulary = json.load(f) 


def tokenizer_fit_process(q: Queue, dataset: IterableDataset, shard_name: str) -> None:
    tokenizer = Tokenizer()
    tokenizer.fit_hugging_face(dataset, shard_name)
    q.put(tokenizer.vocabulary)  

def merge_vocabularies(base: Tokenizer, vocabularies: list[dict]) -> None:
    for vocab in tqdm(vocabularies, desc="Merging vocab"):
        for word, _ in vocab.items():
            if word not in base._found_word:
                base._last_token += 1
                base._vocabulary[word] = base._last_token
                base._found_word.add(word)

def create_tokenizer(
    hugging_face_dataset: str,version: str,split: str,
    padding_lenght: int,not_found_token: int = -1) -> Tokenizer:

    q: Queue = Queue()
    cpu_count: int = int(os.environ.get("cpu_count", "1"))
    tokenizer: Tokenizer = Tokenizer(padding_lenght, not_found_token)
    divided_datasets: list[IterableDataset] = divide_dataset_shards(hugging_face_dataset, version, 
                                                                    split, cpu_count)

    fn: function = partial(tokenizer_fit_process, q=q)

    list_process: list[Process] = [Process(target=fn,kwargs={"dataset": divided_datasets[i], "shard_name": str(i)})
                                   for i in range(cpu_count)]

    for process in tqdm(list_process, desc="Starting processes"):
        process.start()

    vocabularies = [q.get() for _ in range(cpu_count)]

    for process in list_process:
        process.join()  

    merge_vocabularies(tokenizer, vocabularies)

    return tokenizer

if __name__ == "__main__": 
    test = Tokenizer()

    test.upload_tokenizer("test.json")

    print(test.transform(["Hello my name is Gabriel"]))
    print(len(list(test.vocabulary.keys())))