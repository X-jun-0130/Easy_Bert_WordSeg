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
 最后，得到的input_id, input_segment, mask, label为模型的输入。具体程序在data_process.py里。
 
 第三步：构建模型
 bert模型：
 ```
with tf.variable_scope('bert'):
    bert_embedding = modeling.BertModel(config=bert_config,
                                        is_training=True,
                                        input_ids=input_x,
                                        input_mask=mask,
                                        token_type_ids=input_segment,
                                        use_one_hot_embeddings=False)

    embedding_inputs = bert_embedding.get_sequence_output()
is_training=True表示进行finetune,  use_one_hot_embeddings=False表示不使用TPU。
bert_embedding.get_sequence_output()输出数据形式[batch_size,seq_length,hidden_dim],hidden_dim=768
```
crf模型：
```
#bert预训练模型输出形状[batch_size, max_seq_length, hidden_dim]
#进行处理，形状改为[batch_size*max_seq_length, hidden_dim]
#进行全连接，输出结果形状为[batch_size*max_seq_length, pm.num_tags]
#重新转为三维形状，进行crf层
with tf.variable_scope('crf'):
    outputs = embedding_inputs
    hidden_size = outputs.shape[-1].value
    output = tf.reshape(outputs, [-1, hidden_size])
    output = tf.layers.dense(output, pm.num_tags)
    output = tf.contrib.layers.dropout(output, pm.keep_prob)
    logits = tf.reshape(output, [-1, pm.seq_length, pm.num_tags])
    log_likelihood, transition_params = crf_log_likelihood(inputs=logits, tag_indices=input_y,
                                                           sequence_lengths=real_sequlength)
```
loss损失函数:
```
with tf.variable_scope('loss'):
    loss = tf.reduce_mean(-log_likelihood) #最大似然取负，使用梯度下降
```
optimizer优化器:
```
with tf.variable_scope('optimizer'):
    num_train_steps = int((length_text) / pm.batch_size * pm.num_epochs)
    num_warmup_steps = int(num_train_steps * 0.1)  # 总的迭代次数 * 0.1 ,这里的0.1 是官方给出的，我直接写过来了
    train_op = optimization.create_optimizer(loss, pm.lr, num_train_steps, num_warmup_steps, False)
 官方提供的 optimization 主要是学习速率可以动态调整，如下面简图，学习速率由小到大，峰值就是设置的lr,然后在慢慢变小，
 整个学习速率，呈现三角形

                 -
               -      -
             -          -
           -                 -
```
获取预训练bert模型中所有的训练参数：
```
# 获取模型中所有的训练参数。
tvars = tf.trainable_variables()
# 加载BERT模型
(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, pm.init_checkpoint)

tf.train.init_from_checkpoint(pm.init_checkpoint, assignment_map)

tf.logging.info("**** Trainable Variables ****")
# 打印加载模型的参数
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
session = tf.Session()
session.run(tf.global_variables_initializer())
```
