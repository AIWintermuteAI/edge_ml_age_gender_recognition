from scipy.io import loadmat
from datetime import datetime
import os
import pandas as pd

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def age_to_groups(age):

    if age > 0 and age <= 10:
        age = 0

    elif age > 10 and age <= 20:
        age = 1

    elif age > 20 and age <= 45:
        age = 2

    elif age > 45 and age <= 60:
        age = 3

    elif age > 60:
        age = 4

    return age

def gender_to_groups(gender):

    if gender == 'female':
        gender = 0
    else:
        gender = 1

    return gender


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(path, _format):

    if _format == "mat":
        d = loadmat(path)
        return d["image"], d["gender"][0], d["age"][0]

    elif _format == "csv":
        d = pd.read_csv(path)
        
        for i, age in d.iterrows():
            d.at[i, 'age'] = age_to_groups(d.at[i, 'age'])
            
        for i, gender in d.iterrows():
            d.at[i, 'gender'] = gender_to_groups(d.at[i, 'gender'])
            
        return d["file"].to_numpy(), d["gender"].to_numpy(), d["age"].to_numpy()


def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass
