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


def get_sentences(line, separator):
    line = re.sub(rf'[^\w.?!;:{separator}]', ' ', line)
    line = re.sub(rf'[.?!;:]+', f'{separator}', line)
    line = re.sub(' +', ' ', line)
    sentences = line.split(f' {separator} ')

    return sentences[:-1]


def get_ngrams(tokenized_text, elmo_embeddings, n=2):
    words = tokenized_text.copy()
    embeddings = list(map(lambda embeddings: np.mean(embeddings, axis=0), elmo_embeddings))

    return zip(nltk.ngrams(words, n), nltk.ngrams(embeddings, n))


def save_dataset(path, dictionary):
    feat_dictionary = defaultdict(list)

    for key in dictionary:
        feat_dictionary[key] = list(np.mean(np.array(dictionary[key]), axis=0))

    feat_dictionary = pd.DataFrame(feat_dictionary).transpose()
    feat_dictionary.to_csv(path)


if __name__ == '__main__':
    main()
