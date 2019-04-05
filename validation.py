import numpy as np

def silhouette_score(clusters, distances):
  a = distances@clusters*clusters/(clusters.sum(axis=0)-1)
  a = np.ma.masked_equal(a, 0, copy=False)
  msk = np.ma.masked_equal(distances@clusters*(1-clusters)/clusters.sum(axis=0), 0, copy=False).min(axis=1)
  b = (clusters.T*msk).T
  s = (b-a)/np.maximum(a,b)
  return s.mean(axis=0).mean()
def purity_score(y_true, y_pred):
  return (y_true.T@y_pred).max(axis=0).sum()/y_pred.sum()
def f1_score(y_true, y_pred):
  contingency = y_true.T@y_pred
  recall = (contingency.T/y_true.sum(axis=0)).T
  precision = contingency/y_pred.sum(axis=0)
  f1_scores = 2/(np.power(precision, -1)+np.power(recall,-1))
  return f1_scores.max(axis=1).mean()
