#encoding='utf-8
import tensorflow as tf
import codecs
from bert import tokenization
from parameters import Parameters as pm
from bert_Wordseg import logits, transition_params, input_segment, input_x, mask, keep_pro, real_sequlength
from tensorflow.contrib.crf import viterbi_decode

session = tf.Session()
session.run(tf.global_variables_initializer())
save_path = tf.train.latest_checkpoint('./checkpoints/bert_wordseg')
saver = tf.train.Saver()
saver.restore(sess=session, save_path=save_path)

tokenizer = tokenization.FullTokenizer(vocab_file=pm.vocab_filename, do_lower_case=False) #加载bert汉字词典
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
    for i in sentence:
        word_list.append(i)
    return word_list

def read_file(filename):
    content = []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        eachline = eachline.strip('\n')
        eachline = eachline.strip(' ')
        word_list = get_word(eachline)
        content.append(word_list)
    return content


def predict(sess, sentence):

    text = get_word(sentence)
    text = text[0:pm.seq_length-2]

    text.insert(0, "[CLS]")
    text.append("[SEP]")
    wordlist = text[1:-1]

    word = []
    for key in text:  # 将不存在bert词典中的字变成[UNK]
        if key not in token_vocab:
            key = '[UNK]'
            word.append(key)
        else:
            word.append(key)

    text2id = tokenizer.convert_tokens_to_ids(word)  # 将字序列 变成 数字序列
    segment = [0] * len(text2id)
    mask_ = [1] * len(text2id)
    seqlength = len(text2id)
    while len(text2id) < pm.seq_length:
        text2id.append(0)
        mask_.append(0)
        segment.append(0)
    assert len(text2id) == pm.seq_length
    assert len(mask_) == pm.seq_length
    assert len(segment) == pm.seq_length

    logits_, transition_params_ = sess.run([logits, transition_params], feed_dict={input_x: [text2id],
                                                                                   input_segment: [segment],
                                                                                   mask: [mask_],
                                                                                   real_sequlength: [seqlength],
                                                                                   keep_pro: 1.0})

    # logit 每个子句的输出值，length子句的真实长度，logit[:length]的真实输出值
    # 调用维特比算法求最优标注序列
    label = []
    for logit, length in zip(logits_, [seqlength]):
        viterbi_seq, _ = viterbi_decode(logit[:length], transition_params_)
        label = [key for key in viterbi_seq]

    #截去前后标记符

    label = label[1:-1]

    #读取分词结果
    result = ''
    for i, key in enumerate(wordlist):
        ids = str(label[i])
        if ids == '4': #'4'表示S
            result += key
            result += ' '
        elif ids == '3':
            result += key
            result += ' '
        else:
            result += key

    return result


if __name__ == '__main__':

    content = read_file(pm.eva_filename)
    for sentence in content:
        print(''.join(sentence))
        pre = predict(session, sentence)
        print(pre)
        print('\n')
