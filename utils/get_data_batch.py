

import time
import random
import torch




def get_batch_info(uid_traj, win_len, uid_mask_day, split=[0.6, 0.1, 0.3]):
    
    batch_all = {'train': [], 'valid': [], 'test': []}

    for uid, traj_all in uid_traj.items():
        traj_num = len(traj_all['loc'])
        valid_num = traj_num - win_len
        num_train = int(valid_num * split[0])
        num_valid = int(valid_num * split[1])
        num_test = valid_num - num_train - num_valid
        batch_all['train'] += [(uid, traj_i) for traj_i in range(0, num_train) if uid_mask_day[uid][traj_i+win_len]==0]
        batch_all['valid'] += [(uid, traj_i) for traj_i in range(num_train, num_train + num_valid) if uid_mask_day[uid][traj_i+win_len]==0]
        batch_all['test'] += [(uid, traj_i) for traj_i in range(valid_num - num_test, valid_num) if uid_mask_day[uid][traj_i+win_len]==0]

    print(f'Trajectory Number: train/valid/test: {len(batch_all["train"])}/{len(batch_all["valid"])}/{len(batch_all["test"])}')

    return batch_all
def get_batch_info_week(uid_traj, win_len, uid_mask_day, split=[0.6, 0.1, 0.3]):
    batch_all = {'train': [], 'valid': [], 'test': []}

    for uid, traj_all in list(uid_traj.items())[:]:
        traj_num = len(traj_all['loc'])
        valid_num = traj_num - win_len*2
        num_train = int(valid_num * split[0])
        num_valid = int(valid_num * split[1])
        num_test = valid_num - num_train - num_valid
        batch_all['train'] += [(uid, traj_i) for traj_i in range(0, num_train) if
                               uid_mask_day[uid][traj_i + win_len] == 0]
        batch_all['valid'] += [(uid, traj_i) for traj_i in range(num_train, num_train + num_valid) if
                               uid_mask_day[uid][traj_i + win_len] == 0]
        batch_all['test'] += [(uid, traj_i) for traj_i in range(valid_num - num_test-7, valid_num-7) if
                              uid_mask_day[uid][traj_i + win_len] == 0]

    print(
        f'Trajectory Number: train/valid/test: {len(batch_all["train"])}/{len(batch_all["valid"])}/{len(batch_all["test"])}')

    return batch_all


def get_data_batch(uid_traj, uid_attr, uid_mask_day, uid_mask_traj, batch_all, params, mode='train'):
    '''
    Get batch data 
    '''
    batch_size, device, win_len = params.batch_size, params.device, params.win_len
    # get all batch
    random.shuffle(batch_all)
    batch_num = round(len(batch_all) / batch_size)
    print('Batch number is ', batch_num)
    
    # generate one batch
    for i in range(batch_num):
        # print info
        if i == 0:
            print('Batch:', end=' ')
        elif batch_num < 100:
            print(i, end=',')
        elif i>0 and i % (batch_num//100) == 0:
            print(f'{i/batch_num*100:.0f}%', end=',')
        if i == batch_num - 1:
            print('Batch End')

        # construct batch
        batch_start, batch_end = i * batch_size, (i+1) * batch_size
        batch_list = batch_all[batch_start:batch_end]

        # construct batch content
        uid_batch = []
        attr_home = []
        attr_work = []
        attr_com = []
        loc_src = []
        tim_src = []
        mask_day = []
        mask_traj = []
        loc_tar = []
        tim_tar = []

        for uid, traj_start in batch_list:
            # day mask
            mask_day_tmp = uid_mask_day[uid][traj_start:traj_start+win_len]
            if sum(mask_day_tmp) > params.max_mask_day:   # drop too much mask
                continue                       # drop this (X, y) in batch
            else:
                mask_day.append(mask_day_tmp)
            # user id
            uid_batch.append(uid)
            # attr
            attr_home.append([uid_attr[uid]['home']])
            attr_com.append([uid_attr[uid]['commuter']])
            attr_work.append([uid_attr[uid]['work']] if uid_attr[uid]['work'] != None else [0])
            # traj source 
            loc_src.append(uid_traj[uid]['loc'][traj_start:traj_start+win_len])
            tim_src.append(uid_traj[uid]['tim'][traj_start:traj_start+win_len])
            mask_traj.append(uid_mask_traj[uid][traj_start:traj_start+win_len])
            # traj target
            loc_tar.append(uid_traj[uid]['loc'][traj_start+win_len])
            tim_tar.append(uid_traj[uid]['tim'][traj_start+win_len])

        # to torch
        uid_batch = torch.LongTensor(uid_batch).to(device)
        attr_batch = (
            torch.LongTensor(attr_home).to(device),
            torch.LongTensor(attr_com).to(device),
            torch.LongTensor(attr_work).to(device))
        loc_src = torch.LongTensor(loc_src).to(device)
        tim_src = torch.LongTensor(tim_src).to(device)
        mask_day = torch.BoolTensor(mask_day).to(device)
        mask_traj = torch.BoolTensor(mask_traj).to(device)
        loc_tar = torch.LongTensor(loc_tar).to(device)
        tim_tar = torch.LongTensor(tim_tar).to(device)

        # return
        yield (uid_batch, attr_batch, loc_src, tim_src, mask_day, mask_traj), (loc_tar, tim_tar)
def get_data_batch_week(uid_traj, uid_attr, uid_mask_day, uid_mask_traj, batch_all, params, mode='train'):
    '''
    Get batch data
    '''
    batch_size, device, win_len = params.batch_size, params.device, params.win_len
    # get all batch
    random.shuffle(batch_all)
    batch_num = round(len(batch_all) / batch_size)
    print('Batch number is ', batch_num)

    # generate one batch
    for i in range(batch_num):
        # print info
        if i == 0:
            print('Batch:', end=' ')
        elif batch_num < 100:
            print(i, end=',')
        elif i > 0 and i % (batch_num // 100) == 0:
            print(f'{i / batch_num * 100:.0f}%', end=',')
        if i == batch_num - 1:
            print('Batch End')

        # construct batch
        batch_start, batch_end = i * batch_size, (i + 1) * batch_size
        batch_list = batch_all[batch_start:batch_end]

        # construct batch content
        uid_batch = []
        attr_home = []
        attr_work = []
        attr_com = []
        loc_src = []
        tim_src = []
        mask_day = []
        mask_traj = []
        loc_tar = []
        loc_tar_week=[]
        tim_tar = []
        tim_tar_week = []

        for uid, traj_start in batch_list:
            # day mask
            mask_day_tmp = uid_mask_day[uid][traj_start:traj_start + win_len]
            if sum(mask_day_tmp) > params.max_mask_day:  # drop too much mask
                continue  # drop this (X, y) in batch
            else:
                mask_day.append(mask_day_tmp)
            # user id
            uid_batch.append(uid)
            # attr
            attr_home.append([uid_attr[uid]['home']])
            attr_com.append([uid_attr[uid]['commuter']])
            attr_work.append([uid_attr[uid]['work']] if uid_attr[uid]['work'] != None else [0])
            # traj source
            loc_src.append(uid_traj[uid]['loc'][traj_start:traj_start + win_len])
            tim_src.append(uid_traj[uid]['tim'][traj_start:traj_start + win_len])
            mask_traj.append(uid_mask_traj[uid][traj_start:traj_start + win_len])
            # traj target
            loc_tar.append(uid_traj[uid]['loc'][traj_start + win_len])
            loc_list=[]
            tim_list=[]
            count=0
            idx=traj_start + win_len
            while count<7:
                if idx<len(uid_mask_day[uid]):
                    if uid_mask_day[uid][idx]!=1:
                        loc_list.append(uid_traj[uid]['loc'][idx])
                        tim_list.append(uid_traj[uid]['tim'][idx])
                        count+=1
                        idx += 1
                    else:
                        loc_list.append(uid_traj[uid]['loc'][traj_start + win_len])
                        tim_list.append(uid_traj[uid]['tim'][traj_start + win_len])
                        count += 1
                        idx += 1
                else:
                    break
            loc_tar_week.append(loc_list)
            tim_tar_week.append(tim_list)
            tim_tar.append(uid_traj[uid]['tim'][traj_start + win_len])

        # to torch
        uid_batch = torch.LongTensor(uid_batch).to(device)
        attr_batch = (
            torch.LongTensor(attr_home).to(device),
            torch.LongTensor(attr_com).to(device),
            torch.LongTensor(attr_work).to(device))
        loc_src = torch.LongTensor(loc_src).to(device)
        tim_src = torch.LongTensor(tim_src).to(device)
        mask_day = torch.BoolTensor(mask_day).to(device)
        mask_traj = torch.BoolTensor(mask_traj).to(device)
        loc_tar = torch.LongTensor(loc_tar).to(device)
        tim_tar = torch.LongTensor(tim_tar).to(device)

        # return
        yield (uid_batch, attr_batch, loc_src, tim_src, mask_day, mask_traj), (loc_tar, tim_tar,loc_tar_week,tim_tar_week)