from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
dir(housing)
# print(housing.feature_names)
# print(housing.data.shape)

import pandas as pd

## Panel Data Set 面板数据，Excel数据，方便python编程的数据

dataframe = pd.DataFrame(housing.data, columns=housing.feature_names)
# 添加房价
dataframe['price'] = housing['target']
# print(dataframe)

## 什么因素对房价影响是最大的
# 这里相关性分析，是如果>1那么正相关，成比例增加的
# 这里面可以看横轴是变量x，纵轴是因变量y，看x和y之间的关系，这里可以看最后一个列price，看不同的类别对price的相关性影响
correlation = dataframe.corr()
print(correlation)
