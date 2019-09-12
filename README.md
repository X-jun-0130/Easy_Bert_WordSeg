# Easy_Bert_WordSeg
利用Bert_CRF进行中文分词

# 预训练的bert中文模型：
下载地址链接：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
![存储为如此形式：](https://github.com/NLPxiaoxu/Easy_Bert_classify/blob/master/image/bert_model.png)

## checkpoints 里面保存的参数太大了,一个有1G，上传不了. 有需要的自己训练。重要的是了解模型搭建的过程。

## 备注
模型训练了10个epoch，我就停掉了。原本的打算是每个epoch结束，返回一次测试集精度，精度提升才保存模型(跟之前的bert文本分类做法一样), 但精度测试这里有点问题，没找到好的方法，所以就直接每5个epoch保存一次模型。10个epoch后，loss从40 ---> 1左右。继续迭代下去能到0.1左右。

# 模型过程
## 数据集
数据集来自nlpcc2016, 数量有点少 20000 条左右。

## 数据处理
 第一步： 将句子转化为 'B M E S' 序列
 B：表示词的开始
 M：表示词的中间
 E：表示词的结束
 S：表示单字成词
 例如：中文模型 --->  BMME
 ```
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
 ```
 第二步：处理成bert模型所需数据
```
tokenizer = tokenization.FullTokenizer(vocab_file=pm.vocab_filename, do_lower_case=False) 加载预训练bert模型的中文词典
text = tokenizer.tokenize(eachline)  将句子转换成 字列表，如：输入“你好”，返回['你','好']
bert需要在字列表首位添加 "[CLS]"，尾部添加"[SEP]"字符
text.insert(0, "[CLS]")
text.append("[SEP]")
返回数据为：['CLS','你','好','SEP']
然后将列表字转换成数字，还是利用bert中文字典，
text2id = tokenizer.convert_tokens_to_ids(text) 将字列表 变成 数字列表

segemnt表示输入的句子是段落几，第一段落用0表示，第二段落用1表示，...。bert能够接受的中文句子长度为512，大于这个长度可以分段输入。

mask矩阵，句子原长度部分，权重值为1，padding得来的部分，权重值为0

接下来，将label转换成数字，state_list = {'B': 1, 'M': 2, 'E': 3, 'S': 4, '[CLS]': 5, '[SEP]': 6}
_label = [state_list[key] for key in label_]
        if len(text2id) != len(_label):
            print(i)
padding部分，不足设定程度的句子，补0
        while len(text2id) < pm.seq_length:
            text2id.append(0)
            mask_.append(0)
            segment.append(0)
            _label.append(0)
        assert len(text2id) == pm.seq_length
        assert len(mask_) == pm.seq_length
        assert len(segment) == pm.seq_length
        assert len(_label) == pm.seq_length
```
 
