import os
import pandas as pd
import rampwf as rw
import numpy as np
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image, ImageOps

def index_tuple_generator(lst):
    return dict([(i, x) for i, x in enumerate(lst)])

problem_title = 'Pokemon type classification'

path_images = os.path.join("data", "Images")
path_types = os.path.join("data", "Table")

df=pd.read_parquet(os.path.join(path_types, "Pokemon_name_and_type.parquet"))
image_list = os.listdir(path_images)
dic = {}
for image_name in image_list:
    index=image_name.split("_")[0]
    img = Image.open(os.path.join(path_images, image_name))
    numpy_array = np.array(img)
    dic[index]=numpy_array
df["Image"]=df.index.map(dic)

targets = ["Type 1", "Type 2"]

X = df.drop(columns=targets)
y = df[targets].apply(lambda x: [x['Type 1']] if pd.isna(x['Type 2']) else [x['Type 1'], x['Type 2']], axis=1)
mlb = MultiLabelBinarizer()
y = pd.DataFrame(mlb.fit_transform(y), columns=mlb.classes_)

dictionary_types = index_tuple_generator(mlb.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

_target_column_name = 'Types'
_prediction_label_names = list(dictionary_types)
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='acc')
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=42)
    return cv.split(X, y)


def get_train_data(path='.'):
    return X_train, y_train.to_numpy()


def get_test_data(path='.'):
    return X_test, y_test.to_numpy()