class Parameters(object):
    seq_length = 128        #max length of cnnsentence
    num_tags = 7            #number of labels

    keep_prob = 0.9          #droppout
    lr = 0.00005             #learning rate

    num_epochs = 20          #epochs
    batch_size = 12          #batch_size


    train_filename='./data/nlpcc2016-word-seg-train.dat'  #train data
    test_filename='./data/nlpcc2016-wordseg-test.dat'    #test data
    eva_filename = './data/evel.txt'    #eva data
    vocab_filename='./bert_model/chinese_L-12_H-768_A-12/vocab.txt'        #vocabulary
    bert_config_file = './bert_model/chinese_L-12_H-768_A-12/bert_config.json'
    init_checkpoint = './bert_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
