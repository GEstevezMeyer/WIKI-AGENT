import re
import json 

from datasets import load_dataset
from tqdm import tqdm 


class Tokenizer:
    def __init__(self,padding_lenght:int = 512,not_found_token = -1):
        self._vocabulary:dict = {}
        self._found_word:set = set()
        self._last_token:int = 0
        self.padding_lenght = padding_lenght
        self.not_found_token = -1 
        

    @property
    def vocabulary(self):
        return self._vocabulary

    def _cleaning_text(self,text:str) -> list[str]: 
        text:str = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]', '', text).lower()
        word_list:list[str] = text.split(" ")
        print(word_list)
        return  word_list

    def _fit_text(self,text:str) -> None:
        word_list:list[str] = self._cleaning_text(text)
        for word in word_list:
            if (word not in self._found_word) and (len(word) >= 1):
                self._last_token+= 1
                self._vocabulary[word] = self._last_token
                self._found_word.add(word)

    def fit(self,dataset:list[str]) -> None:
        for i in tqdm(range(len(dataset)),f"Fiting text "):
            self._fit_text(dataset[i])

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
        for i in tqdm(range(len(dataset)),desc = "Transforming dataset"):
            res.append(self._transform_text(dataset[i]))
        return res 
    
    def save_tokenizer(self,path:str) -> None:
        #TODO
        pass 
        
    def upload_tokenizer(self,path:str) -> None:
        #TODO
        pass 

if __name__ == "__main__": 
    test = ["Hola como estas? Bien y tu é "]
    test_t = ["HOLA COMO ESTAS? gracias","HOLA COMO ESTAS? gracias eeee"]
    tokenizer = Tokenizer(padding_lenght=10)
    tokenizer.fit(test)
    print(tokenizer.vocabulary)

    print(tokenizer.transform(test_t))
