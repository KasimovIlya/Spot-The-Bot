import re
import torch
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, BertModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = []
    tokenizer = BertTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    model = BertModel.from_pretrained("sberbank-ai/ruBert-base", output_hidden_states=True).to(device)

    with open(rf"russian_lit_corpus.txt", 'r', encoding='UTF-8') as file:
        for index, text in tqdm(enumerate(file)):
            try:
                sentences = get_sentences(text, '[SEP]')
            except Exception:
                continue

            for sentence in sentences:
                tokenized_text, tokens_tensor, segments_tensors = prepare_sentence(sentence, tokenizer,
                                                                                   device=device)
                token_embeddings = get_embeddings(tokens_tensor, segments_tensors, model)
                embeddings.append(token_embeddings)

    embeddings = np.vstack(embeddings)
    np.save(f"embeddings", embeddings)


def get_sentences(line, separator):
    line = re.sub(rf'[^\w.?!;:{separator}]', ' ', line)
    line = re.sub(rf'[.?!;:]+', f'{separator}', line)
    line = re.sub(' +', ' ', line)
    sentences = line.split(f' {separator} ')

    return sentences[:-1]


def prepare_sentence(sentence, tokenizer, device):
    tokenized_text = tokenizer.tokenize(f"[CLS] {sentence} [SEP]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens], device=device)
    segments_tensors = torch.tensor([segments_ids], device=device)

    return tokenized_text, tokens_tensor, segments_tensors


def get_embeddings(tokens_tensor, segments_tensors, model, encoder=None):
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]

    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    if encoder is not None:
        token_embeddings = encoder(token_embeddings)

    return token_embeddings.cpu().detach().numpy()


if __name__ == '__main__':
    main()
