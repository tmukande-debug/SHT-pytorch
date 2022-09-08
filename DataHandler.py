import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import torch
import numpy as np
import pickle
from Params import args


np.random.seed(666)
torch.manual_seed(666)

def transpose(mat):
  coomat = sp.coo_matrix(mat)
  return csr_matrix(coomat.transpose())

def negSamp(temLabel, sampSize, nodeNum):
  negset = [None] * sampSize
  cur = 0
  while cur < sampSize:
    rdmItm = np.random.choice(nodeNum)
    if temLabel[rdmItm] == 0:
      negset[cur] = rdmItm
      cur += 1
  return negset

def transToLsts(mat, mask=False, norm=False):
  shape = torch.Size(mat.shape)
  mat = sp.coo_matrix(mat)
  indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
  data = mat.data
  
  if norm:
    rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
    colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
    for i in range(len(mat.data)):
      row = indices[0, i]
      col = indices[1, i]
      data[i] = data[i] * rowD[row] * colD[col]
  # half mask
  if mask:
    spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
    data = data * spMask

  if indices.shape[0] == 0:
    indices = np.array([[0, 0]], dtype=np.int32)
    data = np.array([0.0], np.float32)

  data = torch.from_numpy(data)
  #a =torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()
  return indices, data, shape

class DataHandler:
  def __init__(self):
    if args.data == 'yelp':
      predir = '/content/drive/MyDrive/SHT/yelp/'
    elif args.data == 'tmall':
      predir = '/content/drive/MyDrive/SHT/tmall/'
    elif args.data == 'gowalla':
      predir = '/content/drive/MyDrive/SHT/gowalla/'
    self.predir = predir
    self.trnfile = predir + 'trnMat.pkl'
    self.tstfile = predir + 'tstMat.pkl'

  def LoadData(self):
    with open(self.trnfile, 'rb') as fs:
      trnMat = (pickle.load(fs) != 0).astype(np.float32)
    # test set
    with open(self.tstfile, 'rb') as fs:
      tstMat = pickle.load(fs)
    tstLocs = [None] * tstMat.shape[0]
    tstUsrs = set()
    for i in range(len(tstMat.data)):
      row = tstMat.row[i]
      col = tstMat.col[i]
      if tstLocs[row] is None:
        tstLocs[row] = list()
      tstLocs[row].append(col)
      tstUsrs.add(row)
    tstUsrs = np.array(list(tstUsrs))

    self.trnMat = trnMat
    self.tstLocs = tstLocs
    self.tstUsrs = tstUsrs
    args.edgeNum = len(trnMat.data)
    args.user, args.item = self.trnMat.shape