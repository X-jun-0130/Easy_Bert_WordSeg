#encoding:utf-8
import numpy as np
import re
import codecs
from bert import tokenization
from parameters import Parameters as pm

state_list = {'B': 1, 'M': 2, 'E': 3, 'S': 4, '[CLS]': 5, '[SEP]': 6}

#读取bert中字作为列表
token_vocab = []
with codecs.open(pm.vocab_filename, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_vocab.append(token)

#将句子转换为字序列
def get_word(sentence):
    word_list = []
    sentence = ''.join(sentence.split(' '))
    for i in sentence:
        word_list.append(i)
    return word_list

#将句子转换为BMES序列
def get_str(sentence):
    output_str = []
    sentence = re.sub('  ', ' ', sentence) #发现有些句子里面，有两格空格在一起
    list = sentence.split(' ')
    for i in range(len(list)):
        if len(list[i]) == 1:
            output_str.append('S')
        elif len(list[i]) == 2:
            output_str.append('B')
            output_str.append('E')
        else:
            M_num = len(list[i]) - 2
            output_str.append('B')
            output_str.extend('M'* M_num)
            output_str.append('E')
    return output_str

def read_file(filename):
    content, label = [], []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        eachline = eachline.strip('\n')
        eachline = eachline.strip(' ')
        word_list = get_word(eachline)
        letter_list = get_str(eachline)
        content.append(word_list)
        label.append(letter_list)
    return content, label

content, label = read_file(pm.train_filename)
length_text = len(content)

def Token(filename):
    input_id, input_segment, mask, label = [], [], [], []
    tokenizer = tokenization.FullTokenizer(vocab_file=pm.vocab_filename, do_lower_case=False) #加载bert汉字词典

    content, labels = read_file(filename)
    for i in range(len(content)):
        eachline = content[i]
        eachline = eachline[0:pm.seq_length-2]
        wordlist= []
        for key in eachline: #将不存在bert词典中的字变成[UNK]
            if key not in token_vocab:
                key = '[UNK]'
                wordlist.append(key)
            else:
                wordlist.append(key)

        label_ = labels[i][0:pm.seq_length-2]
        #text = tokenizer.tokenize(eachline) #将句子变成 字序列
        text = wordlist
        text.insert(0, "[CLS]")
        text.append("[SEP]")
        text2id = tokenizer.convert_tokens_to_ids(text) #将字序列 变成 数字序列
        segment = [0] * len(text2id)
        mask_ = [1] * len(text2id)
        label_.insert(0, "[CLS]")
        label_.append("[SEP]")

        _label = [state_list[key] for key in label_]
        if len(text2id) != len(_label):
            print(i)

        while len(text2id) < pm.seq_length:
            text2id.append(0)
            mask_.append(0)
            segment.append(0)
            _label.append(0)
        assert len(text2id) == pm.seq_length
        assert len(mask_) == pm.seq_length
        assert len(segment) == pm.seq_length
        assert len(_label) == pm.seq_length

        input_id.append(text2id)
        input_segment.append(segment)
        mask.append(mask_)
        label.append(_label)
    return input_id, input_segment, mask, label

# input_id, input_segment, mask, label = Token(pm.train_filename)
# print(input_id[3])
# print(label[3])


def batch_iter(id, segment, mask, label, batch_size = pm.batch_size):
    data_len = len(id)
    num_batch = int((data_len - 1)/batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    id = np.array(id)
    segment = np.array(segment)
    mask = np.array(mask)
    label = np.array(label)
    '''
    np.arange(4) = [0,1,2,3]
    np.random.permutation([1, 4, 9, 12, 15]) = [15,  1,  9,  4, 12]
    '''
    id_shuff = id[indices]
    segment_shuff = segment[indices]
    mask_shuff = mask[indices]
    label_shuff = label[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield id_shuff[start_id:end_id], segment_shuff[start_id:end_id], mask_shuff[start_id:end_id], label_shuff[start_id:end_id]


def sequence(x_batch):
    seq_len = []
    for line in x_batch:
        length = np.sum(np.sign(line))
        seq_len.append(length)

    return seq_len