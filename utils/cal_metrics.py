

import time
from scipy.stats import entropy as KL
import numpy as np
import itertools
import geobleu
from collections import Counter

def Geo_Belu(sync,true,loccor):
    traj_nyc_7 = sync
    traj_target_7 = true
    id2loc=loccor
    traj_nyc = traj_nyc_7
    traj_target = traj_target_7
    count = 0
    belu_score = 0
    for k in list(traj_nyc.keys())[:]:
        seqs = traj_nyc[k]
        for s in range(len(seqs)):
            traj = traj_nyc[k][s]
            belu_list_sync = [(id2loc[traj[i]][0], id2loc[traj[i]][1]) if traj[i] != 0 else (0, 0) for i in
                              range(len(traj))]
            tar_loc_list = traj_target[k][s]
            belu_list_true = [
                (id2loc[tar_loc_list[i]][0], id2loc[tar_loc_list[i]][1]) if tar_loc_list[i] != 0 else (0, 0) for i in
                range(len(tar_loc_list))]
            belu = geobleu.calc_geobleu_orig(belu_list_sync, belu_list_true)
            belu_score += belu
            count += 1
    return belu_score/count
def dist_error_func(sync_traj,true_traj,loccor):
    from math import radians, cos, sin, asin, sqrt
    def get_HaversineDistance(lon_lat1, lon_lat2):
        '''
        Info:
            calculate haversine distance with two coordinates.
        '''
        lon1, lat1 = lon_lat1
        lon2, lat2 = lon_lat2
        # map longitude and latitude to radian
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        # delta coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        # distance
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371
        distance = round(distance, 3)
        return distance

    def geo_distance_error(traj_sync, traj_true):
        loccor[0] = loccor[1]
        distance_sync = [get_HaversineDistance((loccor[traj_sync[i]][1], loccor[traj_sync[i]][0]),
                                               (loccor[traj_sync[i + 1]][1], loccor[traj_sync[i + 1]][0])) for i in
                         range(len(traj_sync) - 1)]
        distance_true = [get_HaversineDistance((loccor[traj_true[i]][1], loccor[traj_true[i]][0]),
                                               (loccor[traj_true[i + 1]][1], loccor[traj_true[i + 1]][0])) for i in
                         range(len(traj_true) - 1)]
        dist_error = abs(sum(distance_sync) - sum(distance_true))
        return dist_error

    def travel_dist_error():
        dist_error_list = []
        for k in sync_traj.keys():
            seqs_sync = sync_traj[k]
            seqs_true = true_traj[k]
            for i in range(len(seqs_sync)):
                seq_sync = seqs_sync[i]
                seq_true = seqs_true[i]
                dist_error = geo_distance_error(seq_sync, seq_true)
                dist_error_list.append(dist_error)
        return np.mean(dist_error_list)
    dist_error=travel_dist_error()
    return dist_error




def cal_metrics(uid_traj_sync, uid_traj_true, loc_coor, bin_size=0.5):
    '''calculate metrics:
    one travel behavior related: Loc_ACC, Distance_Error
    user related (pattern/distribution):  JSD_ChainLen, JSD_TravelDist, JSD_DepartTime (dailyBehavior)
    '''

    # define variables
    Loc_Acc_count = [0, 0]
    Distance_Error_list = []
    JSD_TravelDist_list = []

    # iterate user
    for uid, traj_sync_all in uid_traj_sync.items():
        # metrics related
        travelDist_sync, travelDist_true = [], []

        # true data
        traj_true_all = uid_traj_true[uid]
        for idx, traj_sync in enumerate(traj_sync_all):
            traj_true = traj_true_all[idx]
            # directly trajectory 
            # location accuracy
            pred_true = np.array(traj_sync) == np.array(traj_true)
            Loc_Acc_count[0] += pred_true.sum()
            Loc_Acc_count[1] += len(traj_true)

            # distance error
            for idx, flag in enumerate(pred_true):
                if flag:
                    Distance_Error_list.append(0)
                else:
                    loc_sync, loc_true = traj_sync[idx], traj_true[idx]
                    coor_sync, coor_true = loc_coor[loc_sync], loc_coor[loc_true]
                    Distance_Error_list.append(get_HaversineDistance(coor_sync, coor_true))
        
            # chain-related metrics
            # get unique chain
            chain_metric = get_chain_metric(traj_sync, loc_coor)
            travelDist_sync.append(chain_metric)
            chain_metric = get_chain_metric(traj_true, loc_coor)
            travelDist_true.append(chain_metric)

        # calculate JSD for each user
        JSD_TravelDist_list.append(calculate_jsd(travelDist_sync, travelDist_true, bin_size=bin_size))

    # single travel related metrics 
    loc_acc = Loc_Acc_count[0] / Loc_Acc_count[1]

    dist_error = np.mean(Distance_Error_list)

    geo_bleu=Geo_Belu(uid_traj_sync, uid_traj_true, loc_coor)

    return (loc_acc, dist_error, geo_bleu)



def get_chain_metric(traj, loc_coor):

    # get unique chain
    loc_chain, tim_chain = get_unique_chain(traj)

    # daily travel distance
    travelDist = 0
    for idx in range(len(loc_chain)-1):
        travelDist += get_HaversineDistance(loc_coor[loc_chain[idx]], loc_coor[loc_chain[idx+1]])

    return travelDist


def get_unique_chain(traj):
    loc_chain, tim_chain = [], []
    for key, group in itertools.groupby(traj):
        loc_chain.append(key)
        tim_chain.append(len(list(group)))
    loc_chain = np.array(loc_chain)
    tim_chain = np.array(tim_chain)
    return loc_chain, tim_chain



from math import radians, cos, sin, asin, sqrt
def get_HaversineDistance(lon_lat1, lon_lat2):
    '''
    Info:
        calculate haversine distance with two coordinates.
    '''
    lon1, lat1 = lon_lat1
    lon2, lat2 = lon_lat2
    # map longitude and latitude to radian 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2]) 
    # delta coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # distance
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2 * asin(sqrt(a)) * 6371
    distance = round(distance, 3)
    return distance 


def calculate_jsd(arr_sync, arr_true, bin_size, log_scale=False):
    '''
    Calculate Jensen-Shanon Divergence of two list
    '''
    # calculate bins
    bin_max, bin_min = max(max(arr_sync), max(arr_true)), min(min(arr_sync), min(arr_true), 0)
    if log_scale:
        max_bin = np.round(np.log10(bin_max), 1) + bin_size
        bins = np.power(10, np.arange(0, max_bin+bin_size, bin_size))
        bins = np.insert(bins, 0, bin_min)
    else:
        bins = np.arange(bin_min, bin_max+bin_size, bin_size)

    # get distribution of the list
    p1, bins = get_distribution(arr_true, bins)
    p2, _ = get_distribution(arr_sync, bins)

    # calculate JSD
    m = (p1+p2) / 2
    jsd = 0.5 * (KL(p1, m) + KL(p2, m))
    return jsd



def get_distribution(arr, bins):
    '''
    Calculate distribution with bins
    Params:
        arr: 1 dimension array or list
        bin_size: int
        bins: array, bin edges
    '''
    # get distribution
    p, _ = np.histogram(arr, bins)
    if np.sum(p) == 0:
        print(p)
    p = p / np.sum(p)
    # fill nan  (nan due to out of bin range)
    p = np.nan_to_num(p)

    return p, bins