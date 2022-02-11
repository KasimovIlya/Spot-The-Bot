import re
import nltk
import torch
import pathlib
import numpy as np
import pandas as pd
import tqdm

from auto_encoder.auto_encoder import AutoEncoder
from elmoformanylangs import Embedder
from collections import defaultdict


def main():
    # Выбираем место для работы с тензорами.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dictionary = defaultdict(list)

    # Загружаем предобученный ELMo.
    elmo = Embedder(rf'model', batch_size=16)

    # Загружаем предобученный автоэнкодер.
    encoder = AutoEncoder(input_dim=1024, hidden_dim=8)
    encoder.load_state_dict(torch.load(rf"auto_encoder\auto_encoder_params"))
    encoder.to(device)
    encoder.test()

    # Размеры датасетов в результате работы.
    dataset_lengths = sorted([10, 20, 40, 100, 250, 500, 1000, 2000, 5000], reverse=True)

    with open(rf"..\russian_lit_corpus.txt", 'r', encoding='UTF-8') as file:
        for index, text in tqdm.tqdm(enumerate(file), total=dataset_lengths[0]):
            try:
                if index == dataset_lengths[-1]:
                    save_dataset(rf"..\..\..\_datasets\russian\ELMo_{index}.csv", dictionary)
                    dataset_lengths.pop()
                    if len(dataset_lengths) == 0:
                        break
            except IOError as error:
                print(f"{index}: {error}")
                continue

            # Разбиваем текст на слова/токены и закидываем в ELMo.
            tokenized_text = list(map(lambda sentence: sentence.split(' '), get_sentences(text, '[SEP]')))
            text_embeddings = elmo.sents2elmo(tokenized_text)

            # Отдельно обрабатываем каждое предложение автоэнкодером и извлекаем n-граммы, затем записываем их в словарь.
            for tokenized_sentence, sentence_embeddings in zip(tokenized_text, text_embeddings):
                sentence_embeddings = torch.FloatTensor(sentence_embeddings).to(device)
                sentence_embeddings = encoder(sentence_embeddings).cpu().detach().numpy()
                
                for key_gram, feats_gram in get_ngrams(tokenized_sentence, sentence_embeddings):
                    dictionary[' '.join(key_gram)].append(np.hstack(feats_gram))


# Очищает лемматизированный текст от стоп-символов и разбивает на предложения по указанному разделителю.
def get_sentences(line: str, separator: str) -> list[str]:
    line = re.sub(rf'[^\w.?!;:{separator}]', ' ', line)
    line = re.sub(rf'[.?!;:]+', f'{separator}', line)
    line = re.sub(' +', ' ', line)
    sentences = line.split(f' {separator} ')

    return sentences[:-1]


# Получает из текста и набора эмбеддингов n-граммы указанного размера.
def get_ngrams(tokenized_text: list, elmo_embeddings: np.ndarray, n=2) -> zip:
    return zip(nltk.ngrams(tokenized_text, n), nltk.ngrams(elmo_embeddings, n))


# Сохраняет датасет n-грамм эмбеддингов в виде .csv файла по указанному пути.
def save_dataset(path: str or pathlib.Path, dictionary: defaultdict[list]):
    feat_dictionary = defaultdict(list)

    for key in dictionary:
        feat_dictionary[key] = list(np.mean(np.array(dictionary[key]), axis=0))

    feat_dictionary = pd.DataFrame(feat_dictionary).transpose()
    feat_dictionary.to_csv(path)


if __name__ == '__main__':
    main()
