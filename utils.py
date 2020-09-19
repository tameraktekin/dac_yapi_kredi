import os
import itertools


def organize_data(train_dir):
    users = []

    for file in os.listdir(train_dir):
        sign_info = file.split('-')[1]
        users.append(sign_info)

    users_2 = users.copy()
    combinations = itertools.combinations(users_2, 2)
    return combinations
