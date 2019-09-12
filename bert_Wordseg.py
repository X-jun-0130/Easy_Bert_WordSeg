from data_process import length_text, sequence
from bert import modeling, optimization
import tensorflow as tf
from parameters import Parameters as pm
import os
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
input_segment = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_segment')
mask = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='mask')
input_y = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_y')
real_sequlength = tf.placeholder(tf.int32, shape=[None], name='seq_length')
keep_pro = tf.placeholder(tf.float32, name='drop_out')

bert_config = modeling.BertConfig.from_json_file(pm.bert_config_file)

with tf.variable_scope('bert'):
    bert_embedding = modeling.BertModel(config=bert_config,
                                        is_training=True,
                                        input_ids=input_x,
                                        input_mask=mask,
                                        token_type_ids=input_segment,
                                        use_one_hot_embeddings=False)

    embedding_inputs = bert_embedding.get_sequence_output()

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

with tf.variable_scope('loss'):
    loss = tf.reduce_mean(-log_likelihood) #最大似然取负，使用梯度下降

with tf.variable_scope('optimizer'):
    num_train_steps = int((length_text) / pm.batch_size * pm.num_epochs)
    num_warmup_steps = int(num_train_steps * 0.1)  # 总的迭代次数 * 0.1 ,这里的0.1 是官方给出的，我直接写过来了
    train_op = optimization.create_optimizer(loss, pm.lr, num_train_steps, num_warmup_steps, False)


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

def feed_data(_ids, _mask, _segment, label, seq_length, keep_prob):
    feet_dict = {input_x: _ids,
                 mask: _mask,
                 input_segment: _segment,
                 input_y: label,
                 real_sequlength: seq_length,
                 keep_pro: keep_prob
                 }
    return feet_dict
