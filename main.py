import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
import json
import numpy as np


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def grabweakClassArr(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

nextItemNum = 1
nextItemFile = 'next1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()

if args.dataset=='AotM':
    fp = open('./data/AotMListItems_len5_item5_cut1000.negativeEvalTest')
    creator_list_dict = grabweakClassArr('./data/AotM_creator_list.dict')
elif args.dataset=='Spotify':
    fp = open('/home/dumengnan/yunhe/code/data/spotifyItems_len20_item20_cut1000.negativeEvalTest')
    creator_list_dict = grabweakClassArr('./data/Spotify_creator_list.dict')
elif args.dataset=='Zhihu':
    fp = open('/home/dumengnan/yunhe/code/data/zhihuListItems_len5_item5_cut1000.negativeEvalTest')
    creator_list_dict = grabweakClassArr('./data/Zhihu_creator_list.dict')
elif args.dataset=='Goodreads':
    fp = open('/home/dumengnan/yunhe/code/data/goodreadsListItems_len5_item5_cut1000.negativeEvalTest')
    creator_list_dict = grabweakClassArr('./data/Goodreads_creator_list.dict')

text = fp.read()
negativeTestDict = json.loads(text)

UniqueCreator = {}
for listId in creator_list_dict:
    creatorId = creator_list_dict[listId]
    UniqueCreator[creatorId] = 1
creatorNum = len(UniqueCreator)
print 'number of creators: ', creatorNum


if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset, nextItemNum)
[train, valid, test, listnum, itemnum] = dataset
num_batch = len(train) / args.batch_size
cc = 0.0
for u in train:
    cc += len(train[u])
print 'average sequence length: %.2f' % (cc / len(train))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(train, listnum, itemnum, creator_list_dict, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(creatorNum, itemnum, args)
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

threshold = 200

#try:
for epoch in range(1, args.num_epochs + 1):

    for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        creator, seq, pos, neg = sampler.next_batch()
        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                {model.creator: creator, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                 model.is_training: True})
    print('epoch: ', epoch, 'auc: ', auc, 'loss: ', loss)
    
    if epoch>threshold and epoch % 3 == 0:
        t1 = time.time() - t0
        T += t1
        'Validation result: '
        ndcg_10_val, recall_10_val = evaluate_valid(model, dataset, args, sess, creator_list_dict)
        print 'epoch:%d, time: %f(s), validation (NDCG@10: %.4f, Recall@10: %.4f)' % (epoch, T, ndcg_10_val, recall_10_val)

        print 'Evaluating',
        t_test = evaluate(model, dataset, args, sess, negativeTestDict, nextItemNum, creator_list_dict)
        
        print ''
#            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
#            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
        print 'epoch:%d, time: %f(s), test (NDCG@1: %.4f, Recall@1: %.4f, Precision@1: %.4f)' % (epoch, T, t_test[0][0], t_test[0][1], t_test[0][2])
        print 'epoch:%d, time: %f(s), test (NDCG@5: %.4f, Recall@5: %.4f, Precision@5: %.4f)' % (epoch, T, t_test[1][0], t_test[1][1], t_test[1][2])
        print 'epoch:%d, time: %f(s), test (NDCG@10: %.4f, Recall@10: %.4f, Precision@10: %.4f)' % (epoch, T, t_test[2][0], t_test[2][1], t_test[2][2])
        print 'epoch:%d, time: %f(s), test (NDCG@20: %.4f, Recall@20: %.4f, Precision@20: %.4f)' % (epoch, T, t_test[3][0], t_test[3][1], t_test[3][2])
        
        f.write('epoch: '+ str(epoch) + ' ' + str(t_test) + '\n')#str(t_valid) + ' ' +
        f.flush()
        t0 = time.time()
# except:
#     sampler.close()
#     f.close()
#     exit(1)

# f.close()
# sampler.close()
# print("Done")
