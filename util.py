import sys, math
import copy
import random
import numpy as np
from collections import defaultdict

def data_partition(fname, nextItemNum):
    listnum = 0
    itemnum = 0
    list_ContainItems_dict = defaultdict(list)
    train = {}
    valid = {}
    test = {}
    # assume list/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        l, i = line.rstrip().split(' ')
        l = int(l)
        i = int(i)
        listnum = max(l, listnum)
        itemnum = max(i, itemnum)
        list_ContainItems_dict[l].append(i)

    for _list in list_ContainItems_dict:
        nfeedback = len(list_ContainItems_dict[_list])
        if nfeedback < 3:
            train[_list] = list_ContainItems_dict[_list]
            valid[_list] = []
            test[_list] = []
        else:
            train[_list] = list_ContainItems_dict[_list][:nfeedback-nextItemNum-1]
            valid[_list] = []
            valid[_list].append(list_ContainItems_dict[_list][-nextItemNum-1])#this validation data is used when we do testing so far actually
            test[_list] = []
            test[_list] = list_ContainItems_dict[_list][-nextItemNum:]

    return [train, valid, test, listnum, itemnum]


def evaluate(model, dataset, args, sess, negativeTestDict, nextItemNum, creator_user_dict):
    
    [train, valid, test, listnum, itemnum] = copy.deepcopy(dataset)
    
    NDCG20 = 0.0
    HT20 = 0.0
    Precision20 = 0.0
    NDCG10 = 0.0
    HT10 = 0.0
    Precision10 = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    Precision5 = 0.0
    NDCG1 = 0.0
    HT1 = 0.0
    Precision1 = 0.0
    valid_list = 0.0
    
    lists = xrange(1, listnum + 1)
    for u in lists:
        
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)#always consider recent maxlen items, if more than maxlen, cut it off, if less, add 0 as paddings.
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        #item_idx = [test[u][0]]
        item_idx = []
        negativeTestItemList = negativeTestDict[str(u-1)]
        for item in reversed(negativeTestItemList):
            item_idx.append(item+1)

        creator = creator_user_dict[u]
        predictions = -model.predict(sess, [creator], [seq], item_idx)
        predictions = predictions[0]

        valid_list += 1
        
        NDCGperList20, HTperList20, precisionList20 = precisionAndRecallNDCG(predictions, test[u], 20, nextItemNum)
        NDCGperList10, HTperList10, precisionList10 = precisionAndRecallNDCG(predictions, test[u], 10, nextItemNum)
        NDCGperList5, HTperList5, precisionList5 = precisionAndRecallNDCG(predictions, test[u], 5, nextItemNum)
        NDCGperList1, HTperList1, precisionList1 = precisionAndRecallNDCG(predictions, test[u], 1, nextItemNum)
        
        NDCG20 += NDCGperList20
        HT20 += HTperList20
        Precision20 += precisionList20
        NDCG10 += NDCGperList10
        HT10 += HTperList10
        Precision10 += precisionList10
        NDCG5 += NDCGperList5
        HT5 += HTperList5
        Precision5 += precisionList5
        NDCG1 += NDCGperList1
        HT1 += HTperList1
        Precision1 += precisionList1
        
        
        if valid_list % 1000 == 0:
            print '.',
            sys.stdout.flush()

    result = [[NDCG1 / valid_list, HT1 / valid_list, Precision1/valid_list],[NDCG5 / valid_list, HT5 / valid_list, Precision5/valid_list],[NDCG10 / valid_list, HT10 / valid_list, Precision10/valid_list],[NDCG20 / valid_list, HT20 / valid_list, Precision20/valid_list]]
    return result

def precisionAndRecallNDCG(predictions, testList, k, nextItemNum):
    NDCGperUser = 0.0
    recallperUser = 0.0
    precisionUser = 0.0
    idcgPerUser = 0.0
    HTperUser = 0
    groundtruthLength = len(testList)
    if groundtruthLength >= k:
        for rank in range(k):
            idcgPerUser += 1 / np.log2(rank + 2)
    else:
        for rank in range(groundtruthLength):
            idcgPerUser += 1 / np.log2(rank + 2)
    
        
    for i in range(groundtruthLength):
        rank = predictions.argsort().argsort()[i]
        
        if rank < k:
            NDCGperUser += 1 / np.log2(rank + 2)
            HTperUser += 1

    NDCGperUser = NDCGperUser*1.0/idcgPerUser
    
    recallperUser = HTperUser*1.0/nextItemNum
    precisionUser = HTperUser*1.0/k
#    if k==5:
#        print nextItemNum, k, HTperUser, precisionUser
    return NDCGperUser, recallperUser, precisionUser

def evaluate_valid(model, dataset, args, sess, creator_user_dict):
    [train, valid, test, listnum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_list = 0.0
    HT = 0.0

    lists = xrange(1, listnum + 1)
    for u in lists:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        creator = creator_user_dict[u]
        predictions = -model.predict(sess, [creator], [seq], item_idx, )
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_list += 1
        
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_list % 1000 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_list, HT / valid_list
