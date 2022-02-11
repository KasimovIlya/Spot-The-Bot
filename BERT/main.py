import re
import nltk
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from auto_encoder.auto_encoder import AutoEncoder
from collections import defaultdict


def main():
    # Выбираем место для работы с тензорами.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dictionary = defaultdict(list)
    
    # Загружаем предобученный BERT.
    tokenizer = BertTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    model = BertModel.from_pretrained("sberbank-ai/ruBert-base", output_hidden_states=True).to(device)

    # Загружаем предобученный автоэнкодер.
    encoder = AutoEncoder(input_dim=768, hidden_dim=8)
    encoder.load_state_dict(torch.load(rf"auto_encoder\auto_encoder_params"))
    encoder.to(device)
    encoder.test()

    # Размеры датасетов в результате работы.
    dataset_lengths = sorted([10, 20, 40, 100, 250, 500, 1000, 2000, 5000], reverse=True)

    with open(rf"..\russian_lit_corpus.txt", 'r', encoding='UTF-8') as file:
        for index, text in tqdm(enumerate(file), total=dataset_lengths[0]):
            try:
                if index == dataset_lengths[-1]:
                    save_dataset(dictionary, rf"..\..\..\_datasets\russian\BERT_{index}.csv")
                    dataset_lengths.pop()
                    if len(dataset_lengths) == 0:
                        break
                sentences = get_sentences(text, '[SEP]')
            except IOError as error:
                print(f"{index}: {error}")
                continue

            for sentence in sentences:
                # Каждое предложение обрабатываем с помощью BERT отдельно от остальных. После используем автоэнкодер и получаем n-граммы, записываем в словарь.
                tokenized_text, tokens_tensor, segments_tensors = prepare_sentence(sentence, tokenizer, device=device)
                token_embeddings = get_embeddings(tokens_tensor, segments_tensors, model, encoder)

                for key_seq, val_seq in get_n_grams(tokenized_text, token_embeddings):
                    dictionary[" ".join(key_seq)].append(np.hstack(val_seq))


# Очищает лемматизированный текст от стоп-символов и разбивает на предложения по указанному разделителю.
def get_sentences(line, separator):
    line = re.sub(rf'[^\w.?!;:{separator}]', ' ', line)
    line = re.sub(rf'[.?!;:]+', f'{separator}', line)
    line = re.sub(' +', ' ', line)
    sentences = line.split(f' {separator} ')

    return sentences[:-1]


# Подготавливает предложение к передаче в BERT.
def prepare_sentence(sentence, tokenizer, device):
    tokenized_text = tokenizer.tokenize(f"[CLS] {sentence} [SEP]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens], device=device)
    segments_tensors = torch.tensor([segments_ids], device=device)

    return tokenized_text, tokens_tensor, segments_tensors


# Получает для каждого токена эмбеддинг с последнего слоя BERT.
def get_embeddings(tokens_tensor, segments_tensors, bert_model, encoder=None):
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]

    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    
    # Если есть энкодер, применяется кодирование.
    if encoder is not None:
        token_embeddings = encoder(token_embeddings)

    return token_embeddings.cpu().detach().numpy()


# Получает из текста и набора эмбеддингов n-граммы указанного размера.
# Из-за особенностей работы токенизатора BERT разбивает некоторые слова на части и для каждой получает вложение. 
# Такие части склеиваются в одно слово, а в соответствие ставится усреднение эмбеддингов.
def get_n_grams(tokenized_text, token_embeddings, n=2):
    words = []
    embeddings = []

    index = 0
    while index < len(tokenized_text):
        word_parts = [tokenized_text[index]]

        start = index
        while index + 1 < len(tokenized_text) and tokenized_text[index + 1].startswith("##"):
            index += 1
            word_parts.append(tokenized_text[index])
        index += 1
        end = index

        word = "".join(word_parts).replace("##", "")
        words.append(word)
        embeddings.append(np.mean(token_embeddings[start:end], axis=0))

    return zip(nltk.ngrams(words[1:-2], n), nltk.ngrams(embeddings[1:-2], n))


# Сохраняет датасет n-грамм эмбеддингов в виде .csv файла по указанному пути.
def save_dataset(dictionary, path):
    output_dict = dict()
    for key in dictionary:
        output_dict[key] = np.mean(np.vstack(dictionary[key]), axis=0)

    dataframe = pd.DataFrame(output_dict).transpose()
    dataframe.to_csv(path)


if __name__ == '__main__':
    main()
