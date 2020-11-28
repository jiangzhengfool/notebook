### 网格搜索
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
clf = KNeighborsClassifier()
parameters = {'n_neighbors': [ 4, 5,6,7,8,9], 'p':[1, 2, 3,4],'weights':['uniform', 'distance'],'algorithm':['auto','ball_tree','kd_tree','brute'], }
#n_jobs =-1使用全部CPU并行多线程搜索
gs = GridSearchCV(clf, parameters, refit = True, cv = 10, verbose = 1, n_jobs = 8)
gs.fit(X,y) #Run fit with all sets of parameters.
print('最优参数: ',gs.best_params_)
print('最佳性能: ', gs.best_score_)
````