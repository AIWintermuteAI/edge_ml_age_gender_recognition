import os
import pandas as pd

ages = ['0-10', '11-20', '21-45', '46-60', '60-100']

def age_to_groups(age):

    for group in ages:
        if age >= int(group.split('-')[0]) and age <= int(group.split('-')[1]):
            age = ages.index(group)

    return age

def gender_to_groups(gender):

    if gender == 'female':
        gender = 0
    else:
        gender = 1

    return gender

def rebalance_data(frame, class_name):
    max_size = frame[class_name].value_counts().max()
    lst = [frame]
    for class_index, group in frame.groupby(class_name):
        lst.append(group.sample(max_size-len(group), replace=True))
    frame_new = pd.concat(lst)

    return frame_new

def load_data(path, rebalance = False):

    d = pd.read_csv(path)
    
    for i, row in d.iterrows():
        d.at[i, 'age'] = age_to_groups(d.at[i, 'age'])
        #d.at[i, 'gender'] = gender_to_groups(d.at[i, 'gender'])

    if rebalance:
        print('Samples before rebalance: {}'.format(len(d)))    
        d = rebalance_data(d, 'age')    
        print('Samples after rebalance: {}'.format(len(d)))

    return d["file"].to_numpy(), d["gender"].to_numpy(), d["age"].to_numpy()


def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass
