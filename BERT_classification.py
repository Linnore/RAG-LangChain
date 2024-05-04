import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig


def extract_embedding(texts, model, tokenizer, first=None, strategy='pooling', device='cpu', max_token_len=512,
                      embedding_size=1024):
    embedding_list = []
    if strategy == 'pooling':
        # print("Using strategy: avg_pooling among all chunks.")
        with torch.no_grad():
            cnt = 0
            for text in texts:
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                num_chunks = (input_ids.shape[1] - 1) // max_token_len + 1

                embedding = torch.zeros((1, embedding_size)).to(device)
                start_pos = 0
                end_pos = 0
                for i in range(num_chunks):
                    start_pos = end_pos
                    if i == num_chunks - 1:
                        end_pos = input_ids.shape[1]
                    else:
                        end_pos += max_token_len
                    if (end_pos - start_pos > 512):
                        print(end_pos, start_pos)
                    # print(start_pos, end_pos)
                    output = model(
                        input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos], output_hidden_states=True)
                    # embedding += output.pooler_output
                    embedding += output.hidden_states[-1][:, 0, :]

                embedding /= num_chunks
                embedding_list.append(embedding.cpu())
                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    elif strategy == 'last':
        # Using the last chunk
        print('Using default strategy: keeping the last chunk.')
        with torch.no_grad():
            cnt = 0
            for text in texts:
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                end_pos = input_ids.shape[1]
                start_pos = max(0, end_pos - max_token_len)

                output = model(
                    input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                # embedding = output.pooler_output
                embedding = output.last_hidden_state[:, 0, :]

                embedding_list.append(embedding.cpu())

                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    elif strategy == 'first':
        # Using the first chunk
        print('Using strategy: keeping the first chunk.')
        with torch.no_grad():
            cnt = 0
            for text in texts:
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                start_pos = 0
                end_pos = min(input_ids.shape[1], max_token_len)

                output = model(
                    input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                # embedding = output.pooler_output
                embedding = output.last_hidden_state[:, 0, :]

                embedding_list.append(embedding.cpu())

                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    else:
        raise Exception("Unsupported strategy!")

    return torch.cat(embedding_list)


def get_id_label_mapping(path):
    id_label_map = pd.read_csv(path)
    id2label = {}
    label2id = {}
    for (id, label) in zip(id_label_map.ID, id_label_map.label):
        label = str(label)
        id2label[id] = label
        label2id[label] = id
    return label2id, id2label


class MLP_C(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP_C, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size)
        self.fc2 = nn.Linear(2 * input_size, 2 * input_size)
        self.fc3 = nn.Linear(2 * input_size, input_size)
        self.fc4 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class MLP_B(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP_B, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size * 0.25))
        self.fc2 = nn.Linear(int(input_size * 0.25), num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class BERT_Classification():
    def __init__(self, bert_model, mlp_c, mlp_b, id_map, strategy):
        self.embedding = None
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_model)
        config = AutoConfig.from_pretrained(bert_model).to_dict()
        self.max_token_len = config['max_position_embeddings'] - 2
        self.embedding_size = config['hidden_size']
        self.id_map = id_map
        label2id, id2label = get_id_label_mapping(self.id_map)
        self.num_classes = len(id2label)
        self.mlp_C = MLP_C(self.embedding_size, self.num_classes)
        self.mlp_C.load_state_dict(torch.load(mlp_c))
        self.strategy = strategy
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.mlp_B = MLP_B(self.embedding_size, 2)
        self.mlp_B.load_state_dict(torch.load(mlp_b))

    def embedded(self, question):
        embedding = extract_embedding(question, model=self.bert_model, tokenizer=self.tokenizer, strategy=self.strategy,
                                      max_token_len=self.max_token_len,
                                      embedding_size=self.embedding_size)
        self.embedding = embedding

    def predict(self):
        label2id, id2label = get_id_label_mapping(self.id_map)
        self.mlp_C.eval()
        with torch.no_grad():
            outputs = self.mlp_C(self.embedding)
            pred = F.sigmoid(outputs[0])
            pred = (pred > 0.5).int()
        pred = np.where(pred)[0]
        # print(pred)

        result = [id2label[idx] for idx in pred]
        print(result)
        return result

    def classify(self):
        self.mlp_B.eval()
        with torch.no_grad():
            outputs = self.mlp_B(self.embedding)
            result = str(torch.argmax(outputs, dim=1).numpy())

        print(result)
        return result


if __name__ == '__main__':
    bert = BERT_Classification(bert_model='/home/yjy/ARIN7102/project/cls_bert/bert_chinese_mc_base', mlp_c='/home/yjy/ARIN7102/project/cls_bert/MLP_classification.pt',
                               mlp_b='/home/yjy/ARIN7102/project/cls_bert/MLP_recognization.pt',
                               id_map='/home/yjy/ARIN7102/project/cls_bert/id_label_mapping.csv', strategy='pooling')
    # bert.predict([
    #     '请问宝宝得了奶癣该怎么医治才好呢?宝宝2个月吃母乳，脸上长了许多小颗粒，底有红红的，有点像书上介绍的奶癣了，听说奶癣处理不当的话，会引起以后不能吃含海鲜还有蛋白质含量高的食品，请问我现在该怎么医治才好呢？非常感谢。'])

    bert.embedded(['我有车'])
    bert.predict()
    bert.classify()

    # bert.predict(['我有头晕'])
    #
    # bert.predict(['我有头晕但不发烧'])
    #
    # bert.predict(['我有头晕并且发烧'])
    #
    # bert.predict(['我感冒发烧'])
    #
    # bert.predict(['我感冒不发烧'])
