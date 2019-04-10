import numpy as np

class Neo_k_medoids():
  def __init__(self, iter_max = 10000, c_max=100, tolerance=0.001):
    """initialize clustering object
    :param max_iter: max number of iterations before stopping if algorithm does not converge
    :param c_max: number of random initializations to test
    :param tolerance: minimum cohesion change to consider convergence
    """
    self.iter_max = iter_max
    self.c_max = c_max
    self.tolerance = tolerance
    self.clusters = None
    self.medoids = None
    self.total_deviation = None
    self.all_total_deviations = []
    self.X = None
  def cluster(self, X, k, alpha=0, beta=0):
    """Generate nonexhaustive overlapping clusters
    :param X: square numpy distance matrix comparing objects
    :param k: number of clusters to generate
    :param alpha: overlap proportion parameter
    :param beta: outlier proportion parameter - maximum proportion of points that may be ignored as outliers

    :constraints:
      k < len(X) : can't have more clusters than data points
      0 <= beta <= 1
      -beta <= alpha << k-1 :
        alpha = -beta indicates no overlap
        alpha = k-1 assigns every point to every cluster

    :returns:
      clusters
      Get clusters, medoids, and total_deviation are saved as attributes with the respective names
    """

    self.X = X
    #get document length
    m,n = X.shape
    if m!=n:
      raise ValueError("X must be a square matrix")
    #check constraints
    if (k > n):
      raise ValueError("k <= len(X) required")
    if (beta < 0 or beta > 1):
      raise ValueError("beta must fall between 0 and 1")
    if (-beta > alpha or alpha > k-1):
      raise ValueError("should have -beta <= alpha <= k-1")

    #iterate c_max times with different initializations to avoid local extremum
    for c in range(self.c_max):
      #randomly initialize cluster medoids
      medoids = np.random.choice(range(n), (k), replace=False)
      #assign and adjust clusters until convergence
      iteration_distances = []
      for iteration in range(self.iter_max):
        object_cluster_distances = X[:, medoids]
        min_distance_order = object_cluster_distances.argsort(axis=None)
        cluster_assignment = np.zeros((n,k))
        assignments = 0

        #initial assignment of points to clusters excluding outliers
        min_row_idx = (np.arange(0, n*k, k) + object_cluster_distances.argmin(axis=1))
        filtered_idxs = min_distance_order[np.isin(min_distance_order, min_row_idx)]
        for index in filtered_idxs:
          if (assignments>=(n-beta*n)):
            break
          #convert flattended index to 2d index
          row = index//k
          col = index%k
          cluster_assignment[row, col] = 1
          assignments += 1
          

        #overlapping assignment of points to clusters
        for index in min_distance_order:
          if (assignments>=(n+alpha*n)):
            break
          row = index//k
          col = index%k
          #prevent an object from being assigned to a cluster already assigned
          if (cluster_assignment[row, col] > 0):
            continue
          cluster_assignment[row, col] = 1
          assignments += 1

        #Swap the medoid with the largest deviation impact
        current_cluster_deviation = (X[:, medoids]*cluster_assignment).sum(axis=0)
        potential_cluster_deviation = X@cluster_assignment
        deviation_improvements = (potential_cluster_deviation-current_cluster_deviation).flatten()
        for improvement in deviation_improvements.argsort():
            point = improvement//k
            cluster = improvement%k
            if deviation_improvements[improvement] > 0:
              break
            if point not in medoids:
              medoids[cluster] = point
              break

        total_deviation = (X[:,medoids]*cluster_assignment).sum()
        iteration_distances.append(total_deviation)

        if len(iteration_distances) >=2:
          improvement = iteration_distances[-2] - iteration_distances[-1]
          #print(improvement)
          if improvement < -10**-5:
            print("WARNING: Negative improvement: ", improvement)
          elif  (improvement <= self.tolerance) and (improvement >=0):
            break
      else:
        try:
          print('WARNING: Algorithm did not converge in', self.iter_max, 'steps')
          print('Last imporvement: ', iteration_distances[-2] - iteration_distances[-1])
        except IndexError:
          pass
      self.all_total_deviations.append(total_deviation)
      if self.clusters is None:
        self.medoids = medoids
        self.clusters = cluster_assignment
        self.total_deviation = total_deviation
      else:
        if total_deviation < self.total_deviation:
          self.medoids = medoids
          self.clusters = cluster_assignment
          self.total_deviation = total_deviation
    return self.clusters

nkm = Neo_k_medoids()
print(nkm.__dict__)
