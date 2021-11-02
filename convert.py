import json
import pandas as pd

data = pd.read_csv("../LOC_val_solution.csv",
                   encoding="cp932",
                   )

data = data.sort_values('ImageId')

f_convert = lambda x: x.split(' ')[0]
data['PredictionString'] = data['PredictionString'].apply(f_convert)

with open('../imagenet_class_index.json') as f:
    class_index = json.load(f)

convert_index = {}
for key,val in class_index.items():
    convert_index[val[0]] = key

f_convert2 = lambda x: convert_index[x]
data['PredictionString'] = data['PredictionString'].apply(f_convert2)

data.to_csv('../convert_class_index.csv', index = False)