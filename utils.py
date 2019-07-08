import numpy as np
import os
import torch
import json
import gzip
import array
import pytrec_eval
import re
import time
import math
import pickle
from joblib import Memory
from itertools import tee
from sklearn.utils import shuffle
from datetime import datetime
from itertools import count
from collections import defaultdict
from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.interactions import Interactions
from spotlight.evaluation import rmse_score

cachedir = '/users/tr.amirhj/weak_recsys/tmp_data'
memory = Memory(cachedir)


# model parameters
RANDOM_SEED = 42
LATENT_DIM = 128
NUM_EPOCHS = 100
BATCH_SIZE = 512
L2 = 1e-6
LEARNING_RATE = 1e-3


@memory.cache
def parse_goodreads(path,
                    split_ratio=0.2,
                    min_explicit_feedbacks=5):

    def parse(path):
        g = gzip.open(path)
        for l in g:
            j = json.loads(l)
            date_added = datetime.strptime(
                re.sub(r"\s+-\d+", "", j['date_added']), '%c')
            date_updated = datetime.strptime(
                re.sub(r"\s+-\d+", "", j['date_updated']), '%c')
            yield(j['user_id'],
                  j['book_id'],
                  bool(j['isRead']),
                  int(j['rating']),
                  time.mktime(date_added.timetuple()),
                  time.mktime(date_updated.timetuple()))

    uid_map = {}
    iid_map = {}

    uids = array.array('i')
    iids = array.array('i')
    ratings = array.array('f')
    timestamps_added = array.array('f')
    timestamps_updated = array.array('f')
    reads = array.array('b')

    # iterate over all records in the dataset
    for (uid, iid, is_read, rate, ts_added, ts_updated) in parse(path):
        # map to continious IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        # update the whole dataset
        uids.append(uid)
        iids.append(iid)
        reads.append(is_read)
        if is_read and rate >= 4:
            # explicit positive feedback
            ratings.append(1)
        elif is_read and rate > 0:
            # explicit negative feedback
            ratings.append(0)
        elif is_read and rate == 0:
            # implicit read feedback
            ratings.append(-1)
        else:
            # implicit shelve feedback
            ratings.append(-2)
        timestamps_added.append(ts_added)
        timestamps_updated.append(ts_updated)

    # convert all interactions to numpy array
    uids = np.array(uids, dtype=np.int32)
    iids = np.array(iids, dtype=np.int32)
    ratings = np.array(ratings, dtype=np.float32)
    timestamps_added = np.array(timestamps_added, dtype=np.float32)
    timestamps_updated = np.array(timestamps_updated, dtype=np.float32)
    reads = np.array(reads, dtype=np.bool)

    print('number of interactions before filtering: %s' %
          format(len(uids), ','))

    # count the number of user's explicit interactions
    users_rates = {}
    for (uid, rate) in zip(uids, ratings):
        if rate > 0:
            num_rates = users_rates.setdefault(uid, 0)
            num_rates += 1
            users_rates[uid] = num_rates

    # find all users which has less than k explicit feedbacks
    mask = array.array('b')
    for uid in uids:
        num_rates = users_rates.setdefault(uid, 0)
        if num_rates <= min_explicit_feedbacks:
            mask.append(False)
        else:
            mask.append(True)
    mask = np.array(mask, dtype=np.bool)

    # clean the dataset
    uids = uids[mask]
    iids = iids[mask]
    ratings = ratings[mask]
    timestamps_added = timestamps_added[mask]
    timestamps_updated = timestamps_updated[mask]
    reads = reads[mask]

    print('number of interactions AFTER filtering: %s' %
          format(len(uids), ','))

    # re-maps the uids and iids into new continious IDs
    uid_map = defaultdict(count().__next__)
    iid_map = defaultdict(count().__next__)
    uids = np.array([uid_map[uid] for uid in uids], dtype=np.int32)
    iids = np.array([iid_map[iid] for iid in iids], dtype=np.int32)

    # shuffle the whole dataset
    uids, iids, ratings, timestamps_added, timestamps_updated, reads = shuffle(uids,
                                                                               iids,
                                                                               ratings,
                                                                               timestamps_added,
                                                                               timestamps_updated,
                                                                               reads)

    # build the shelve dataset
    dataset_shelve = Interactions(user_ids=uids,
                                  item_ids=iids,
                                  ratings=ratings,
                                  timestamps=timestamps_added,
                                  num_users=max(uids)+1,
                                  num_items=max(iids)+1)

    # build the read dataset
    dataset_read = Interactions(user_ids=uids[reads],
                                item_ids=iids[reads],
                                ratings=ratings[reads],
                                timestamps=timestamps_updated[reads],
                                num_users=max(uids)+1,
                                num_items=max(iids)+1)

    # build the rate dataset
    dataset_recommend = Interactions(user_ids=uids[ratings > -1],
                                     item_ids=iids[ratings > -1],
                                     ratings=ratings[ratings > -1],
                                     timestamps=timestamps_updated[ratings > -1],
                                     num_users=max(uids)+1,
                                     num_items=max(iids)+1)

    # split the explicit dataset into train/test
    dataset_recommend_train, dataset_recommend_valid_test = random_train_test_split(
        interactions=dataset_recommend,
        test_percentage=split_ratio)

    # split the test dataset into dev/test keep 1/3 for dev and 2/3 for test
    dataset_recommend_test, dataset_recommend_dev = random_train_test_split(
        interactions=dataset_recommend_valid_test,
        test_percentage=1/2)

    return dataset_recommend_train, dataset_recommend_test, dataset_recommend_dev, dataset_read, dataset_shelve


@memory.cache
def parse_steam(path='/local/terrier/Collections/Recommendations/Steam/',
                split_ratio=0.2,
                min_explicit_feedbacks=5):

    def parse(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield eval(l)

    uid_map = {}
    iid_map = {}

    uids = array.array('i')
    iids = array.array('i')
    ratings = array.array('f')

    for user in parse(path + 'australian_users_items.json.gz'):
        # map to continious user IDs
        uid = uid_map.setdefault(user['user_id'], len(uid_map) + 1)
        for item in user['items']:
            # map to continious item IDs
            iid = iid_map.setdefault(item['item_id'], len(iid_map) + 1)
            play = int(item['playtime_forever'])
            if play > 0:
                # update the play dataset
                uids.append(uid)
                iids.append(iid)
                ratings.append(-1)
            else:
                # update the purchase dataset
                uids.append(uid)
                iids.append(iid)
                ratings.append(-2)

    for user in parse(path + 'australian_user_reviews.json.gz'):
        # map to continious user IDs
        uid = uid_map.setdefault(user['user_id'], len(uid_map) + 1)
        for item in user['reviews']:
            # map to continious item IDs
            iid = iid_map.setdefault(item['item_id'], len(iid_map) + 1)
            recommend = bool(item['recommend'])
            if recommend is True:
                # update the rate dataset
                uids.append(uid)
                iids.append(iid)
                ratings.append(1)
            else:
                # update the rate dataset
                uids.append(uid)
                iids.append(iid)
                ratings.append(0)
            # we're not interested in dislike items now!

    # convert to numpy
    uids = np.array(uids, dtype=np.int32)
    iids = np.array(iids, dtype=np.int32)
    ratings = np.array(ratings, dtype=np.float32)

    print('number of interactions before filtering: %s' %
          format(len(uids), ','))

    # count the number of user's explicit interactions
    users_rates = {}
    for (uid, rate) in zip(uids, ratings):
        if rate > 0:
            num_rates = users_rates.setdefault(uid, 0)
            num_rates += 1
            users_rates[uid] = num_rates

    # find all users which has less than k explicit feedbacks
    mask = array.array('b')
    for uid in uids:
        num_rates = users_rates.setdefault(uid, 0)
        if num_rates <= min_explicit_feedbacks:
            mask.append(False)
        else:
            mask.append(True)
    mask = np.array(mask, dtype=np.bool)

    # clean the dataset
    uids = uids[mask]
    iids = iids[mask]
    ratings = ratings[mask]

    print('number of interactions AFTER filtering: %s' %
          format(len(uids), ','))

    # re-maps the uids and iids into new continious IDs
    uid_map = defaultdict(count().__next__)
    iid_map = defaultdict(count().__next__)
    uids = np.array([uid_map[uid] for uid in uids], dtype=np.int32)
    iids = np.array([iid_map[iid] for iid in iids], dtype=np.int32)

    # shuffle the whole dataset
    uids, iids, ratings = shuffle(uids,
                                  iids,
                                  ratings)

    # build the rate dataset
    dataset_recommend = Interactions(user_ids=uids[ratings > -1],
                                     item_ids=iids[ratings > -1],
                                     ratings=ratings[ratings > -1],
                                     timestamps=np.ones(
                                         len(ratings[ratings > -1])),
                                     num_users=max(uids)+1,
                                     num_items=max(iids)+1)

    # split the explicit dataset into train/test
    dataset_recommend_train, dataset_recommend_valid_test = random_train_test_split(interactions=dataset_recommend,
                                                                                    test_percentage=split_ratio,
                                                                                    random_state=np.random.RandomState(RANDOM_SEED))

    # split the test dataset into dev/test keep 1/3 for dev and 2/3 for test
    dataset_recommend_test, dataset_recommend_dev = random_train_test_split(
        interactions=dataset_recommend_valid_test,
        test_percentage=1/2)

    # build the play dataset
    dataset_play = Interactions(user_ids=uids[ratings > -2],
                                item_ids=iids[ratings > -2],
                                ratings=ratings[ratings > -2],
                                timestamps=np.ones(len(ratings[ratings > -2])),
                                num_users=max(uids)+1,
                                num_items=max(iids)+1)

    # build the shelve dataset
    dataset_purchase = Interactions(user_ids=uids,
                                    item_ids=iids,
                                    ratings=ratings,
                                    timestamps=np.ones(len(ratings)),
                                    num_users=max(uids)+1,
                                    num_items=max(iids)+1)

    return dataset_recommend_train, dataset_recommend_test, dataset_recommend_dev, dataset_play, dataset_purchase


@memory.cache
def parse_douban(path='/local/terrier/Collections/Recommendations/Douban/',
                 split_ratio=0.2,
                 min_explicit_feedbacks=5):

    def parse(path):
        for l in open(path, 'r'):
            yield l.split()

    uid_map = {}
    iid_map = {}

    uids = array.array('i')
    iids = array.array('i')
    ratings = array.array('f')

    for uid, iid, r in parse(path + 'ub.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = int(r)
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        if rate > 4:
            ratings.append(1)
        else:
            ratings.append(0)

    for uid, iid in parse(path + 'uread.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -1
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    for uid, iid in parse(path + 'ureading.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -1
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    for uid, iid in parse(path + 'utag.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -1
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    for uid, iid in parse(path + 'ucomment.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -1
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    for uid, iid in parse(path + 'uwish.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -2
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    # convert to numpy
    uids = np.array(uids, dtype=np.int32)
    iids = np.array(iids, dtype=np.int32)
    ratings = np.array(ratings, dtype=np.float32)

    print('number of interactions before filtering: %s' %
          format(len(uids), ','))

    # count the number of user's explicit interactions
    users_rates = {}
    for (uid, rate) in zip(uids, ratings):
        if rate > 0:
            num_rates = users_rates.setdefault(uid, 0)
            num_rates += 1
            users_rates[uid] = num_rates

    # find all users which has less than k explicit feedbacks
    mask = array.array('b')
    for uid in uids:
        num_rates = users_rates.setdefault(uid, 0)
        if num_rates <= min_explicit_feedbacks:
            mask.append(False)
        else:
            mask.append(True)
    mask = np.array(mask, dtype=np.bool)

    # clean the dataset
    uids = uids[mask]
    iids = iids[mask]
    ratings = ratings[mask]

    print('number of interactions AFTER filtering: %s' %
          format(len(uids), ','))

    # re-maps the uids and iids into new continious IDs
    uid_map = defaultdict(count().__next__)
    iid_map = defaultdict(count().__next__)
    uids = np.array([uid_map[uid] for uid in uids], dtype=np.int32)
    iids = np.array([iid_map[iid] for iid in iids], dtype=np.int32)

    # build the rate dataset
    dataset_recommed = Interactions(user_ids=uids[ratings > 0],
                                    item_ids=iids[ratings > 0],
                                    ratings=ratings[ratings > 0],
                                    num_users=len(uid_map) + 1,
                                    num_items=len(iid_map) + 1)

    # split the explicit dataset into train/test
    dataset_recommed_train, dataset_recommed_test = random_train_test_split(
        interactions=dataset_recommed,
        test_percentage=split_ratio,
        random_state=np.random.RandomState(RANDOM_SEED))

    # split the test dataset into dev/test keep 1/3 for dev and 2/3 for test
    dataset_recommed_test, dataset_recommed_dev = random_train_test_split(
        interactions=dataset_recommed_test,
        test_percentage=1/2)

    # build the intraction (read/reading/wish/tag/comment) dataset
    dataset_read = Interactions(user_ids=uids[ratings > -2],
                                item_ids=iids[ratings > -2],
                                ratings=ratings[ratings > -2],
                                num_users=len(uid_map) + 1,
                                num_items=len(iid_map) + 1)

    # build the wish dataset
    dataset_wish = Interactions(user_ids=uids,
                                item_ids=iids,
                                ratings=ratings,
                                num_users=len(uid_map) + 1,
                                num_items=len(iid_map) + 1)

    return dataset_recommed_train, dataset_recommed_test, dataset_recommed_dev, dataset_read, dataset_wish


@memory.cache
def parse_dianping(path='/local/terrier/Collections/Recommendations/Dianping/',
                   split_ratio=0.2,
                   min_explicit_feedbacks=5):

    def parse(path):
        for l in open(path, 'r'):
            yield l.split()

    uid_map = {}
    iid_map = {}

    uids = array.array('i')
    iids = array.array('i')
    ratings = array.array('f')

    for uid, iid, r in parse(path + 'us.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = int(r)
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        if rate > 4:
            ratings.append(1)
        else:
            ratings.append(0)

    for uid, iid in parse(path + 'utaste.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -1
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    for uid, iid in parse(path + 'uenv.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -1
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    for uid, iid in parse(path + 'userve.txt'):
        # map to continious user IDs
        uid = uid_map.setdefault(uid, len(uid_map) + 1)
        iid = iid_map.setdefault(iid, len(iid_map) + 1)
        rate = -1
        # update the dataset
        uids.append(uid)
        iids.append(iid)
        ratings.append(rate)

    # convert to numpy
    uids = np.array(uids, dtype=np.int32)
    iids = np.array(iids, dtype=np.int32)
    ratings = np.array(ratings, dtype=np.float32)

    print('number of interactions before filtering: %s' %
          format(len(uids), ','))

    # count the number of user's explicit interactions
    users_rates = {}
    for (uid, rate) in zip(uids, ratings):
        if rate > 0:
            num_rates = users_rates.setdefault(uid, 0)
            num_rates += 1
            users_rates[uid] = num_rates

    # find all users which has less than k explicit feedbacks
    mask = array.array('b')
    for uid in uids:
        num_rates = users_rates.setdefault(uid, 0)
        if num_rates <= min_explicit_feedbacks:
            mask.append(False)
        else:
            mask.append(True)
    mask = np.array(mask, dtype=np.bool)

    # clean the dataset
    uids = uids[mask]
    iids = iids[mask]
    ratings = ratings[mask]

    print('number of interactions AFTER filtering: %s' %
          format(len(uids), ','))

    # re-maps the uids and iids into new continious IDs
    uid_map = defaultdict(count().__next__)
    iid_map = defaultdict(count().__next__)
    uids = np.array([uid_map[uid] for uid in uids], dtype=np.int32)
    iids = np.array([iid_map[iid] for iid in iids], dtype=np.int32)

    # build the rate dataset
    dataset_recommend = Interactions(user_ids=uids[ratings > 0],
                                     item_ids=iids[ratings > 0],
                                     ratings=ratings[ratings > 0],
                                     num_users=len(uid_map) + 1,
                                     num_items=len(iid_map) + 1)

    # split the explicit dataset into train/test
    dataset_recommend_train, dataset_recommend_test = random_train_test_split(interactions=dataset_recommend,
                                                                              test_percentage=split_ratio,
                                                                              random_state=np.random.RandomState(RANDOM_SEED))

    # split the test dataset into dev/test keep 1/3 for dev and 2/3 for test
    dataset_recommend_test, dataset_recommend_dev = random_train_test_split(
        interactions=dataset_recommend_test,
        test_percentage=1/2)

    # build the intraction (taste, environment, service) dataset
    dataset_implicit = Interactions(user_ids=uids[ratings > -2],
                                    item_ids=iids[ratings > -2],
                                    ratings=ratings[ratings > -2],
                                    num_users=len(uid_map) + 1,
                                    num_items=len(iid_map) + 1)

    return dataset_recommend_train, dataset_recommend_test, dataset_recommend_dev, dataset_implicit


def train_explicit(train_interactions, valid_interactions, run_name):

    # create a linear representation model
    bilinear = BilinearNet(num_users=train_interactions.num_users,
                           num_items=train_interactions.num_items,
                           embedding_dim=LATENT_DIM,
                           user_embedding_layer=None,
                           item_embedding_layer=None)

    # explicit matrix factorization
    model = ExplicitFactorizationModel(loss='regression',
                                       representation=bilinear,
                                       embedding_dim=LATENT_DIM,
                                       n_iter=1,
                                       learning_rate=LEARNING_RATE,
                                       batch_size=BATCH_SIZE,
                                       l2=L2,
                                       random_state=np.random.RandomState(
                                           RANDOM_SEED),
                                       use_cuda=torch.cuda.is_available())

    if os.path.isfile(cachedir + '/' + run_name):
        model = torch.load(cachedir + '/' + run_name)
    else:
        # train the model and monitor valid loss on the test set
        last_rmse = 10000
        for epoch in range(NUM_EPOCHS):
            # fit the model for one epoch
            print('epoch %d start at: ' % (epoch+1),
                  time.asctime(time.localtime(time.time())))
            model.fit(interactions=train_interactions, verbose=False)
            print('epoch %d end at: ' % (epoch+1),
                  time.asctime(time.localtime(time.time())))
            rmse = rmse_score(model, valid_interactions)
            print('RMSE: {:.4f}'.format(rmse))
            if rmse - last_rmse < 0:
                # we got a lower rmse, so continue!
                last_rmse = rmse
                # store the model for future use
                torch.save(model, cachedir + '/' + run_name)
            else:
                break

    return model


def annotate(interactions, model, run_name):
    if os.path.isfile(cachedir + '/' + run_name):
        # load from the disk
        with open(cachedir + '/' + run_name, 'rb') as f:
            interactions = pickle.load(f)
        return interactions
    else:
        # ask the user preference model to predict the rating value for all user/item pairs
        for idx, (u, i, r) in enumerate(zip(interactions.user_ids,
                                            interactions.item_ids,
                                            interactions.ratings)):
            if r < 0:
                r_star, _ = model.predict(
                    user_ids=u, item_ids=np.array([i, 0]))
                interactions.ratings[idx] = r_star
        # store on the disk for future use
        with open(cachedir + '/' + run_name, 'wb') as f:
            pickle.dump(interactions, f)
        return interactions


def evaluate(interactions, model, topk, raw_results=False):
    # create qrel
    qrel = {}
    for (u, i, r) in zip(interactions.user_ids,
                         interactions.item_ids,
                         interactions.ratings):
        u = str(u)
        i = str(i)
        if u not in qrel:
            qrel[u] = {}
        qrel[u][i] = int(r)
    # relevance evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, pytrec_eval.supported_measures)
    # create run
    run = {}
    for uid in np.unique(interactions.user_ids):
        predictions = -model.predict(user_ids=uid)
        predictions_argsort = predictions.argsort()[:topk]
        if str(uid) not in run:
            run[str(uid)] = {}
        for iid in predictions_argsort:
            run[str(uid)][str(iid)] = float(-predictions[iid])

    results = evaluator.evaluate(run)

    mrr = [(uid, query_measures['recip_rank'])
           for uid, query_measures in results.items()]
    ndcg = [(uid, query_measures['ndcg'])
            for uid, query_measures in results.items()]
    ndcg10 = [(uid, query_measures['ndcg_cut_10'])
              for uid, query_measures in results.items()]
    ndcg_5 = [(uid, query_measures['ndcg_cut_5'])
              for uid, query_measures in results.items()]
    mmap = [(uid, query_measures['map'])
            for uid, query_measures in results.items()]
    success_5 = [(uid, query_measures['success_5'])
                 for uid,  query_measures in results.items()]
    success_10 = [(uid, query_measures['success_10'])
                  for uid, query_measures in results.items()]

    if raw_results is False:
        mrr = pytrec_eval.compute_aggregated_measure(
            measure='recip_rank',
            values=[e[1] for e in mrr])
        ndcg = pytrec_eval.compute_aggregated_measure(
            measure='ndcg',
            values=[e[1] for e in ndcg])
        ndcg10 = pytrec_eval.compute_aggregated_measure(
            measure='ndcg_cut_10',
            values=[e[1] for e in ndcg10])
        ndcg_5 = pytrec_eval.compute_aggregated_measure(
            measure='ndcg_cut_5',
            values=[e[1] for e in ndcg_5])
        mmap = pytrec_eval.compute_aggregated_measure(
            measure='map',
            values=[e[1] for e in mmap])
        success_5 = pytrec_eval.compute_aggregated_measure(
            measure='success_5',
            values=[e[1] for e in success_5])
        success_10 = pytrec_eval.compute_aggregated_measure(
            measure='success_10',
            values=[e[1] for e in success_10])

    return mrr, ndcg, ndcg10, ndcg_5, mmap, success_10, success_5


def train_implicit_negative_sampling(train_interactions, valid_interactions, run_name):
    # create a linear representation model
    bilinear = BilinearNet(num_users=valid_interactions.num_users,
                           num_items=valid_interactions.num_items,
                           embedding_dim=LATENT_DIM,
                           user_embedding_layer=None,
                           item_embedding_layer=None)

    # build the implicit model
    model = ImplicitFactorizationModel(loss='bpr',
                                       representation=bilinear,
                                       embedding_dim=LATENT_DIM,
                                       n_iter=1,
                                       learning_rate=LEARNING_RATE,
                                       batch_size=BATCH_SIZE,
                                       l2=L2,
                                       random_state=np.random.RandomState(
                                           RANDOM_SEED),
                                       use_cuda=torch.cuda.is_available())

    if os.path.isfile(cachedir + '/' + run_name):
        model = torch.load(cachedir + '/' + run_name)
    else:
        # train the model and monitor valid loss on the validation set
        latest_mrr = -10000
        for epoch in range(NUM_EPOCHS):
            print('epoch %d start at: ' % (epoch+1),
                  time.asctime(time.localtime(time.time())))
            model.fit(train_interactions)
            print('epoch %d end at: ' % (epoch+1),
                  time.asctime(time.localtime(time.time())))
            mrr, ndcg, ndcg10, ndcg_5, mmap, success_10, success_5 = evaluate(
                valid_interactions, model, 20)
            print('MRR: {:.4f}'.format(mrr))
            if mrr - latest_mrr > 0:
                # we got a better model, so continue!
                latest_mrr = mrr
                # store the model on disk for future use
                torch.save(model, cachedir + '/' + run_name)
            else:
                break

    return model


def most_common(L, k):
    d = defaultdict(int)  # means default value is 0

    for x in L:  # go through list
        d[x] += 1  # increment counts

    # sort dict items by value (count) in descending order
    sorted_items = sorted(d.items(), key=lambda i: i[1], reverse=True)

    # extract the keys
    sorted_keys = [k for k, v in sorted_items]

    # take k best
    return sorted_keys[:k]


def arp(train_interactions, test_interactions, model, topk=20):

    # number of times item i has been rated in the training set
    phi = defaultdict(int)  # means default value is 0

    for x in train_interactions.item_ids:
        phi[x] += 1  # increment counts

    ap = 0

    for uid in np.unique(test_interactions.user_ids):
        predictions = -model.predict(user_ids=uid)
        predictions_argsort = predictions.argsort()[:topk]
        uap = 0
        for iid in predictions_argsort:
            uap += phi[iid]
        ap += uap / len(predictions_argsort)

    ap = ap / len(np.unique(test_interactions.item_ids))

    return ap
