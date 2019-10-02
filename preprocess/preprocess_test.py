import numpy as np
import pandas as pd
import sys
from io import StringIO

# example of reading raw string in a df using StringIO
csv_str = '''a,b,c,d
1.,3.,3.3,
,,5.,6.
2.3,5.6,2.1,1'''

df = pd.read_csv(StringIO(csv_str))
print(df)

#equivalent numpy array
csv_arr = df.values
print(csv_arr)

# contnuous feature 
# feature/sample removal based on nulls

# ------DROPPING FEATURES IN CASE OF MISSING VALUES--------
# dropna variants used
# axis=0 -> remove rows with NaN; axis=1 -> remove columns with NaN
# thresh=k-> drop rows with NaN > k
# subset=[k]-> only drop rows with NaN in column k
# how='all'-> drop rows with all NaN i.e. all features are nulls
print(df.dropna(axis=0))
print(df.dropna(axis=1))
print(df.dropna(how='all'))
print(df.dropna(thresh=3)) # keep rows with at least 3 values
print(df.dropna(subset=['c'])) #drop rows with NaN in c

# ------IMPUTING FEATURE VALUES IN CASE OF MISSING VALUES--------
# use scikit transformer class Imputer
# all scikit tranformers have fit() and transform(); new data point to be transformed should be of same feature size
# scikit model classes like LogisticRegression belong to estimator class; also have fit(), additionally have predict() for supervised learning, can also have transform()
# fit_transform()-> shortcut for calling fit() and transform() on the same array  
# Imputer parameters: strategy=['mean', 'median', 'most_frequent']; axis=[0(along columns), 1(along rows)]; missing_values=NaN
from sklearn.preprocessing import Imputer
impr = Imputer(strategy='mean', missing_values='NaN', axis=0)
impr = impr.fit(df)
df = impr.transform(df)
print(df)

# categorical features
df = pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'], 
['blue', 'XL', 15.3, 'class1']])
# nominal->color;ordinal->size;continuous->price
df.columns = ['color', 'size', 'price', 'label'] 
print(df)

# use df['ordinal_col'].map(order_dict) with a dict representing order between ordinal values
# for class label encoding use map() or sklearn transformer LabelEncoder
# to inverse to original values use sklearn inverse_transform() or in map() by defining a reverse v: [k] dict
# for nominal features you want to avoid any imlicit order which may be understood by h due to different numeric values; solution is one-hot encoding
# one-hot ecoding is taking all possible values for a nominal feature and representing them as integers
# can use sklearn transformer OneHotEncoder or pandas get_dummies()

# mapping ordinal features using dict
size_dict = {
    'XL': 3,
    'L': 2,
    'M': 1
}
inv_size_dict = {v:k for k,v in size_dict.items()}
df['size'] = df['size'].map(size_dict)
print(df['size'].map(inv_size_dict))

# mapping class labels using dict
class_label_dict = {
    'class1': 0,
    'class2': 1
}
df['label'] = df['label'].map(class_label_dict)
print(df['label'].map(class_label_dict))
print('*******\n')
# using sklearn LabelEncoder transformer
from sklearn.preprocessing import LabelEncoder
color_le = LabelEncoder()
print(df)
X = df[['color', 'size', 'price']].values
y = color_le.fit_transform(X[:,0])
# to get back original label
print(color_le.inverse_transform(y))
X[:,0] = color_le.fit_transform(X[:,0])
print(X)


# cant use LabelEncoder for nominal features since it implies some false ordering between values
# use sklearn transformer OneHotEncoder or pandas get_dummies
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

# using pandas get_dummies()
print(df)
print(pd.get_dummies(df[['color']]))
print(pd.get_dummies(df[['size', 'color', 'price']]))
print(pd.get_dummies(df[['size', 'color', 'price']], drop_first=True))