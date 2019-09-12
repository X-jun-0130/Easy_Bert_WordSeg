# Easy_Bert_WordSeg
利用Bert_CRF进行中文分词

# 预训练的bert中文模型：
下载地址链接：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
![存储为如此形式：](https://github.com/NLPxiaoxu/Easy_Bert_classify/blob/master/image/bert_model.png)

## checkpoints 里面保存的参数太大了,一个有1G，上传不了.

模型训练了10个epoch，我就停掉了。原本的打算是每个epoch结束，返回一次测试集精度，精度提升才保存模型(跟之前的bert文本分类做法一样), 但精度测试这里有点问题，没找到好的方法，所以就直接每5个epoch保存一次模型。10个epoch后，loss从40 ---> 1左右。继续迭代下去能到0.1左右。
