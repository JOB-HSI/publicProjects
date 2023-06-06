import os
import numpy as np
import torch
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import word_tokenize,pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def read_data():
    with open(os.path.join(".","医学文本数据集","medical_data" + ".dat"),encoding="utf-8") as f:
        all_data = f.read().split("\n")
    return all_data

def split_textsAndLabels(dataset):
    sentences= []
    texts = []
    labels = []
    for data in dataset:
        if data:
            l,t = data.split("\t")
            # t, l = data.split(" ")
            labels.append(int(l)-1)
            texts.append(t)
            sentences.append(t)
    return labels,texts


def built_curpus(train_texts,embedding_num):
    word_2_index = {"<PAD>":0,"<UNK>":1}
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义标点符号列表
    stops = set(stopwords.words("english")) #停用词

    for text_sentence in train_texts:
        paragraph = text_sentence.lower()       #全部转化成小写字母
        text_words = word_tokenize(paragraph)  # 分词
        text_words = [word for word in text_words if word not in interpunctuations]  # 去除标点符号
        text_words = [word for word in text_words if word not in stops]  #判断分词在不在停用词列表内

        for word in text_words:
            word_2_index[word] = word_2_index.get(word,len(word_2_index))

    print(nn.Embedding(len(word_2_index),embedding_num))
    return word_2_index,nn.Embedding(len(word_2_index),embedding_num)

class TextDataset(Dataset):
    def __init__(self,all_text,all_label,word_2_index,max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self,index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_label[index])

        text_idx = [self.word_2_index.get(i,1) for i in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))

        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)

        return  text_idx,label


    def __len__(self):
        return len(self.all_text)

class Block(nn.Module):
    def __init__(self,kernel_s,embeddin_num,max_len,hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1,out_channels=hidden_num,kernel_size=(kernel_s,embeddin_num)) #  1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
                            # 输入in_channels    输出out_channels    卷积核大小kernel_size
                            #
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len-kernel_s+1))

    def forward(self,batch_emb): # 1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
        c = self.cnn.forward(batch_emb)
        a = self.act.forward(c)
        a = a.squeeze(dim=-1)
        m = self.mxp.forward(a)
        m = m.squeeze(dim=-1)
        return m


class TextCNNModel(nn.Module):
    def __init__(self,emb_matrix,max_len,class_num,hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]

        self.block1 = Block(2,self.emb_num,max_len,hidden_num)
        self.block2 = Block(3,self.emb_num,max_len,hidden_num)
        self.block3 = Block(4,self.emb_num,max_len,hidden_num)
        self.block4 = Block(5, self.emb_num, max_len, hidden_num)
        self.block5 = Block(6, self.emb_num, max_len, hidden_num)
        # self.block6 = Block(7, self.emb_num, max_len, hidden_num)
        # 卷积核宽度不变，高分别是2，3，4

        self.emb_matrix = emb_matrix

        self.classifier = nn.Linear(hidden_num*5,class_num)  # 2 * 3
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,batch_idx,batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)
        b1_result = self.block1.forward(batch_emb)
        b2_result = self.block2.forward(batch_emb)
        b3_result = self.block3.forward(batch_emb)
        b4_result = self.block4.forward(batch_emb)
        b5_result = self.block5.forward(batch_emb)
        # b6_result = self.block6.forward(batch_emb)

        feature = torch.cat([b1_result,b2_result,b3_result,b4_result,b5_result],dim=1) # 1* 6 : [ batch * (3 * 2)]
        pre = self.classifier(feature)

        if batch_label is not None:
            loss = self.loss_fun(pre,batch_label)
            return loss
        else:
            return torch.argmax(pre,dim=-1)


if __name__ == "__main__":
    all_data=read_data();
    train_set, test_set = train_test_split(all_data, test_size=0.3, random_state=42)        #random_state可以使生成的数据集每次都是相同的
    train_label,train_text = split_textsAndLabels(train_set)
    dev_label,dev_text = split_textsAndLabels(test_set)

    embedding = 50
    max_len = 300
    batch_size = 200
    epoch = 100
    lr = 0.01
    hidden_num = 16
    class_num = len(set(train_label))
    # print(torch.cuda.is_available())
    device = "cuda:0"

    word_2_index,words_embedding = built_curpus(train_text,embedding)
    print(word_2_index,words_embedding)
    pass

    train_dataset = TextDataset(train_text,train_label,word_2_index,max_len)
    train_loader = DataLoader(train_dataset,batch_size,shuffle=False)

    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, max_len)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=False)


    model = TextCNNModel(words_embedding,max_len,class_num,hidden_num).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=lr)

    print(words_embedding)

    for e in range(epoch):
        for batch_idx,batch_label in train_loader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_idx,batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")

        right_num = 0
        for batch_idx,batch_label in dev_loader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_idx)
            right_num += int(torch.sum(pre==batch_label))

        print(f"acc = {right_num/len(dev_text)*100:.2f}%")
