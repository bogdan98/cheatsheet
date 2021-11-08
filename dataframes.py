import pandas as pd

1. Descriptive statistics and grouping

data.col.describe # used to describe data series, excluding NaN values
grouped = data[['col1','col2']].iloc[:len(data)].groupby('col3') # grouped by col3
print(grouped.agg(['min', 'max', 'mean', 'median', 'std'])) # descriptive statistics of grouped 

# more grouping
grouped = data.iloc[:len(data)].groupby(['col1', 'col2'])
grouped_median = grouped.median()
grouped_median = grouped_median.reset_index()[['col1','col2', 'col3']]

2. Converting to a table for LaTeX

print(data.to_latex())

3. Replacing/deleting/sorting

data['col'] = data['col'].replace(['a', 'b'], 'c')
data.drop(['col'], axis = 1, inplace = True) # drop a column
data.drop([0,1,2], axis = 0, inplace = True) # drop a row
# default axis = 0

data.sort_values(axis = 0, by = ['col1', 'col2', ..], inplace = True)
# can be sorted by index/list of index or column/list of columns

4. Missing values

data['col'] = data['col'].fillna(value)
data.dropna(axis, how='any', thresh=None, subset=None, inplace=False)

5. Adding a new column based on existing columns

data['col2'] = data['col1'].apply(lambda x: 1 if smth else 2)
data['col2'] = data.apply(lambda x: x['col1']*x['col0'] if ... else ...)

6. Joining and merging dataframes

df1.merge(df2, left_on='lkey', right_on='rkey', suffixes = ('_1', '_2')) # merging on lkey column (can be a list of columns) 
# in df1 and rkey column (can be a list of columns) in df2
df1.merge(df2, how='inner', on='a') # inner merging on column a, using intersection of keys (i.e. intersection of indexes)

df1.join(df2) # works best for joining columsn of df1 and df2 on index when columns in df1 and df2 are different

pd.concat([df1, df2], axis=0, ignore_index = True, join = 'outer') # concatenate dataframes/series along a specified axis 


7. One-hot encoding

one_hot = pd.get_dummies(data[train_col])
data = data.join(one_hot)

8. Transforming the data

### Scaling ###

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xt = scaler.fit_transform(data)

### Box-Cox transform - make data closer to normal ###

from scipy import stats

xt, _ = stats.boxcox(data, lmbda=None, alpha=None)

### Binning the data ###

pd.qcut(data['col'], 4, duplicates = drop)
# find the ranges for bins 1,2,3,4 ~= numerical to ordinal

9. Dataframes from dictionaries

# create a dataframe from a simple dictionary
data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
pd.DataFrame.from_dict(data)

# create a dataframe from a multi-level dictionary,

user_dict = {
   12: {
      'Category 1': {
         'att_1': 1,
         'att_2': 'whatever'
      },
      'Category 2': {
         'att_1': 23,
         'att_2': 'another'
      }
   },
   15: {
      'Category 1': {
         'att_1': 10,
         'att_2': 'foo'
      },
      'Category 2': {
         'att_1': 30,
         'att_2': 'bar'
      }
   }
}

pd.DataFrame.from_dict({
      (i, j): user_dict[i][j]
      for i in user_dict.keys()
      for j in user_dict[i].keys()
   },
   orient = 'index')
   
# when converted to excel, it would look like a table with 
# columns 12, 15 and subcolumns Category 1 and category 2 under
# each of 12, 15. 

10. Pickle format for saving and reading

dataframe.to_pickle('./data.pkl') 
a = pd.read_pickle('./data.pkl', allow_pickle = True)


11. Statistical tests

from scipy.stats import normaltest # for normality
from scipy.stats import shapiro # for normality
from scipy.stats import ks_2samp # similarity of distributions based on 2 samples
from scipy.stats import ttest_ind # ttest for mean values