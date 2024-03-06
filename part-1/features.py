import pandas as pd

import csv
import pickle as pkl

from fasttext import FastText

model = FastText.load_model('cc.en.300.bin')
print(dir(model))

class Embeddings:
    def __init__(self, model) -> None:
        self.model = model
    def get_embeddings(self, words):
        return [self.model.get_word_vector(i) for i in words]
    