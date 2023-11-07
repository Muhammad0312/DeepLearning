import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    def __init__(self):
        # np.random.seed(100)
        self.minx = 0
        self.maxx = 10
        self.miny = 0
        self.maxy = 10
        self.mean = np.random.random_sample(2) * np.array([(self.maxx - self.minx), (self.maxy- self.miny)])
        self.eigvalx = (np.random.random_sample()*(self.maxx - self.minx)/5)**2
        self.angle = np.random.random_sample()*np.pi*2
        self.rotaion_matrix = np.array([[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]])
        self.covariance = np.dot(np.dot(np.transpose(self.rotaion_matrix), np.diag([self.eigvalx, self.eigvalx])), self.rotaion_matrix)


    def get_sample(self, n):
        self.n = n
        self.data = np.random.multivariate_normal(self.mean, self.covariance, self.n)
        return self.data

def sample_gauss_2d(C, N):
    '''
    Generates C gaussian distributions and N samples from each 2D Gaussian distribution

    Arguments
    C: number of gaussians
    N: number of generated samples per gaussian

    Returns
    X: generated samples (C * N) x 2
    Y: true labels (C * N) x 1
    '''
    X = np.empty((0, 2))
    Y = np.empty((0, 1))

    for i in range(C):
        G = Random2DGaussian()
        g = G.get_sample(N)

        X = np.append(X, g, axis=0)
        Y = np.append(Y, np.ones((N, 1)) * i, axis=0)

    return X, Y

def eval_perf_binary(Y,Y_):
    tp = sum(np.logical_and(Y==Y_, Y_==True))
    fn = sum(np.logical_and(Y!=Y_, Y_==True))
    tn = sum(np.logical_and(Y==Y_, Y_==False))
    fp = sum(np.logical_and(Y!=Y_, Y_==False))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp+fn + tn+fp)
    return accuracy, recall, precision

def eval_AP(ranked_labels):
  """Recovers AP from ranked labels"""
  n = len(ranked_labels)
  pos = sum(ranked_labels)
  neg = n - pos
  tp = np.copy(pos)
  tn = 0
  fn = 0
  fp = neg
  
  sumprec=0
  #IPython.embed()
  for x in ranked_labels:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)    

    if x:
      sumprec += precision

    #print (x, tp,tn,fp,fn, precision, recall, sumprec)
    #IPython.embed()

    tp -= x
    fn += x
    fp -= not x
    tn += not x

  
  return sumprec/pos

def graph_data(X,Y_, Y, special=[]):
  """Creates a scatter plot (visualize with plt.show)

  Arguments:
      X:       datapoints
      Y_:      groundtruth classification indices
      Y:       predicted class indices
      special: use this to emphasize some points

  Returns:
      None
  """
  colors = ['b','g','r','c','m','y','k','w']
  markers = ['o','s','D','v','^','x','*','+']
  classification = (Y_ == Y)
  correctly_classified_dp = np.empty((0, 2))
  correctly_classified_labels = np.empty((0, 1))
  incorrectly_classified_dp = np.empty((0, 2))
  incorrectly_classified_labels = np.empty((0, 1))
  
  for dp, y, c in zip(X, Y, classification):
    if c:
       correctly_classified_dp = np.append(correctly_classified_dp, [dp], axis=0)
       correctly_classified_labels = np.append(correctly_classified_labels, [y], axis=0)
    else:
        incorrectly_classified_dp = np.append(incorrectly_classified_dp, [dp], axis=0)
        incorrectly_classified_labels = np.append(incorrectly_classified_labels, [y], axis=0)
  
  if len(special) == 0:
    plt.scatter(correctly_classified_dp[:,0], correctly_classified_dp[:,1], c=[colors[int(k)] for k in correctly_classified_labels], marker='o')
    plt.scatter(incorrectly_classified_dp[:,0], incorrectly_classified_dp[:,1], c=[colors[int(k)] for k in incorrectly_classified_labels], marker='s')
  else:
    plt.scatter(X[:,0], X[:,1], c=[colors[int(k)] for k in Y])
    plt.scatter(X[special,0], X[special,1], c=[colors[int(k)] for k in Y[special]], s=50, marker='D')
plt.show()


def graph_surface(function, rect, offset=0.5, width=256, height=256):
  """Creates a surface plot (visualize with plt.show)

  Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
              ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

  Returns:
    None
  """

  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  #get the values and reshape them
  values=function(grid).reshape((width,height))
  
  # fix the range and offset
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  # draw the surface and the offset
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval)
    
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])

if __name__=="__main__":
    # G=Random2DGaussian()
    # X=G.get_sample(100)
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()
    np.random.seed(100)
    X, Y = sample_gauss_2d(3, 3)
    plt.scatter(X[:,0], X[:,1])
    plt.show()


