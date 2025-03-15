import re
import numpy as np
import pandas as pd
from utils import LinearRegression0
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('insurance.csv')
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['sex'] = data['sex'].map({'male': 1, 'female': 0})

data['region'] = LabelEncoder().fit_transform(data['region'])


# regions = data['region']
# NSRegion = [] #N:1
# WERegion = [] # W:1
# for region in regions: 
#     if 'north' in region:
#         NSRegion.append(1)
#     else:
#         NSRegion.append(0)
#     if 'west' in region:
#         WERegion.append(1)
#     else:
#         WERegion.append(0)
# data.insert(0, 'north', NSRegion)
# data.insert(0, 'west', WERegion)


bmi_max = data['bmi'].max()
bmi_min = data['bmi'].min()
data['bmi'] = (data['bmi'] - bmi_min)/(bmi_max - bmi_min) 
age_max = data['age'].max()
age_min = data['age'].min()
data['age'] = (data['age'] - age_min)/(age_max - age_min)
#data['children'] = StandardScaler().fit_transform(data[['children']])

data, test = train_test_split(data, test_size= 0.2, random_state= 42)

print(data.head())
y = data['charges']
data = data.drop(columns=['charges'])

test_y =  test['charges']
test_x =  test.drop(columns=['charges'])

# print(data.keys())
#print(data.head())


H_Ins_model = LinearRegression0()
H_Ins_model.fit(data, np.log(y))

print(H_Ins_model.weights)
