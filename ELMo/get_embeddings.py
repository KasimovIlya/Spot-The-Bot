import re
import nltk
import numpy as np
import pandas as pd

from elmoformanylangs import Embedder
from tqdm import tqdm
from collections import defaultdict


def main():
    elmo = Embedder(rf'model', batch_size=16)

    embeddings = []

    with open(rf"..\russian_lit_corpus.txt", 'r', encoding='UTF-8') as file:
        for index, text in tqdm(enumerate(file)):
            try:
                tokenized_text = list(map(lambda sentence: sentence.split(' '), get_sentences(text, '[SEP]')))
                token_embeddings = elmo.sents2elmo(tokenized_text)
                embeddings.append(np.vstack(token_embeddings))
            except Exception as exception:
                print(f"{index}: {exception}")

    embeddings = np.vstack(embeddings)
    np.save(f"embeddings", embeddings)


# Очищает лемматизированный текст от стоп-символов и разбивает на предложения по указанному разделителю.
def get_sentences(line, separator):
    line = re.sub(rf'[^\w.?!;:{separator}]', ' ', line)
    line = re.sub(rf'[.?!;:]+', f'{separator}', line)
    line = re.sub(' +', ' ', line)
    sentences = line.split(f' {separator} ')

    return sentences[:-1]

if __name__ == '__main__':
    main()
