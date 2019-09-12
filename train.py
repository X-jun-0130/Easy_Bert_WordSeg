from data_process import Token, batch_iter
from bert_Wordseg import *
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

best = 0.0
n = 0

def predict(sess, x_, x_segment, x_mask):
    seq_length = len([key for key in x_ if key != 0])
    logits_, transition_params_ = sess.run([logits, transition_params], feed_dict={input_x: [x_],
                                                                                   mask: [x_mask],
                                                                                   input_segment: [x_segment],
                                                                                   real_sequlength: [seq_length],
                                                                                   keep_pro: 1.0})
    label_ = []
    for logit, length in zip(logits_, [seq_length]):
        # logit 每个子句的输出值，length子句的真实长度，logit[:length]的真实输出值
        # 调用维特比算法求最优标注序列
        viterbi_seq, _ = viterbi_decode(logit[:length], transition_params_)
        label_ = [key for key in viterbi_seq]
    return label_

print("Loading Training data...")
input_id, input_segment_, mask_, label = Token(pm.train_filename)

test_id, test_segment, test_mask, test_label = Token(pm.eva_filename)

def evaluate(sess, test_id, test_segment, test_mask, test_label):
    A = 1e-10
    leng = 0
    for i in range(len(test_label)):
        pre_lab = predict(sess, test_id[i], test_segment[i], test_mask[i])
        result = test_label[i][0:len(pre_lab)]
        leng += len(pre_lab)
        for w in range(len(result)):
            if result[w] == pre_lab[w]:
                A += 1

    P = A / float(leng)

    return P

tensorboard_dir = './tensorboard/bert_wordseg'
save_dir = './checkpoints/bert_wordseg'
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

tf.summary.scalar("loss", loss)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(tensorboard_dir)
saver = tf.train.Saver()
writer.add_graph(session.graph)

for epoch in range(pm.num_epochs):
        print('Epoch:', epoch + 1)
        num_batchs = int((len(label) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(input_id, input_segment_, mask_, label, pm.batch_size)
        for x_id, x_segment, x_mask, y_label in batch_train:
            n += 1
            sequ_length = sequence(x_id)
            feed_dict = feed_data(x_id, x_mask, x_segment, y_label, sequ_length, pm.keep_prob)
            _,  train_summary, train_loss = session.run([train_op, merged_summary, loss], feed_dict=feed_dict)

            if n % 100 == 0:
                print('步骤:', n, '损失值:', train_loss)
        # P = evaluate(session, test_id, test_segment, test_mask, test_label)
        # print('测试集准确率:', P)
        # if P > best:
        #     best = P
        if (epoch+1) % 5 == 0:
            print("Saving model...")
            saver.save(session, save_path, global_step=((epoch + 1) * num_batchs))
