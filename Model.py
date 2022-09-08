import torch
import torch.nn as nn
from Params import args

torch.manual_seed(666)

class FC(nn.Module):
  def __init__(self, inputdim, outputdim, bias = False):
    super(FC, self).__init__()
    initializer = nn.init.xavier_normal_
    self.W_fc = nn.Parameter(initializer(torch.empty(inputdim, outputdim).cuda())) # shape latdim * latdim
    self.bias = False
    if bias is True:
      initializer = nn.init.zeros_
      self.bias_fc = nn.Parameter(initializer(torch.empty(outputdim).cuda()))
      self.bias = bias

  def forward(self, ret, act = None):
    ret = ret @ self.W_fc
    if self.bias is True:
      ret = ret + self.bias_fc
    if act == 'leakyrelu':
      ret = torch.maximum(args.leaky * ret, ret)
    if act == 'sigmoid':
      ret = torch.sigmoid(ret)
    return ret

class propagate(nn.Module):
  def __init__(self):
    super(propagate, self).__init__()
    initializer = nn.init.xavier_normal_

    self.fc1 = FC(args.hyperNum,args.hyperNum)
    self.fc2 = FC(args.hyperNum,args.hyperNum)

  def forward(self, V, lats, key, hyper):
    lstlat = torch.reshape(lats[-1] @ V, [-1, args.att_head, args.latdim // args.att_head])
    lstlat = torch.permute(lstlat, (1,2,0)) #shape head_num * (latdim/head_num) * (user num or item num)
    temlat1 = lstlat @ key # shape head_num * (latdim/head_num) * (latdim/head_num)
    hyper = torch.reshape(hyper, [-1, args.att_head, args.latdim // args.att_head])
    hyper = torch.permute(hyper, (1,2,0)) #shape head_num * (latdim/head_num) * hyperNum
    temlat1 = torch.reshape(temlat1 @ hyper, [args.latdim, -1]) #shape latdim * hyperNum
    temlat2 = self.fc1(temlat1, act = 'leakyrelu') + temlat1 #shape latdim * hyperNum
    temlat3 = self.fc2(temlat2, act = 'leakyrelu') + temlat2 #shape latdim * hyperNum

    preNewLat = torch.reshape(torch.transpose(temlat3, 0, 1) @ V, [-1, args.att_head, args.latdim//args.att_head]) #shape hyperNum * head_num * (latdim/head_num)
    preNewLat = torch.permute(preNewLat, (1,0,2)) #shape head num * hyperNum * latdim/head_num
    preNewLat = hyper @ preNewLat #shape head_num * (latdim/head_num) * (latdim/head_num)
    newLat = key @ preNewLat #shape head_num * user num or item num * (latdim/head_num)
    newLat = torch.reshape(torch.permute(newLat,(1,0,2)),[-1,args.latdim]) #shape user num or item num * latdim
    lats.append(newLat)

class meta(nn.Module):
  def __init__(self):
    super(meta, self).__init__()
    self.fc1 = FC(args.latdim, args.latdim * args.latdim, bias = True)
    self.fc2 = FC(args.latdim, args.latdim, bias = True)
    #self.actFunc = nn.LeakyReLU(negative_slope=args.leaky)

  def forward(self, hyper, key):
    hyper_mean = torch.mean(hyper, dim=0, keepdim=True) #1 * latdim
    W1 = self.fc1(hyper_mean, act = None)  # 1 * (latdim*latdim)
    W1 = torch.reshape(W1, [args.latdim, args.latdim]) # latdim * latdim
    b1 = self.fc2(hyper_mean, act = None) # 1 * latdim
    ret = key @ W1 + b1
    ret = torch.maximum(args.leaky * ret, ret) # (batchsize * latdim) * (latdim * latdim) + 1*latdim = batchsize * latdim // 534564 ?, 32
    return ret

class SHT(nn.Module):
  def __init__(self, adj, tpadj):
    super(SHT, self).__init__()

    initializer = nn.init.xavier_normal_
    self.adj = adj # user * item
    self.tpadj = tpadj # item * user

    self.uEmbed_ini = nn.Parameter(initializer(torch.empty(args.user,args.latdim).cuda())) #shape user * latdim
    self.iEmbed_ini = nn.Parameter(initializer(torch.empty(args.item,args.latdim).cuda())) #shape item * latdim
    self.uHyper = nn.Parameter(initializer(torch.empty(args.hyperNum,args.latdim).cuda())) #shape hyper num * latdim
    self.iHyper = nn.Parameter(initializer(torch.empty(args.hyperNum,args.latdim).cuda())) #shape hyper num * latdim

    #BUG! only one <K> and one <V> is needed.

    self.K = nn.Parameter(initializer(torch.empty(args.latdim, args.latdim).cuda())) # shape latdim * latdim
    self.V = nn.Parameter(initializer(torch.empty(args.latdim, args.latdim).cuda())) # shape latdim * latdim

    self.user_propagate = nn.ModuleList()
    self.item_propagate = nn.ModuleList()

    self.reg = []

    for i in range(args.gnn_layer):
      self.user_propagate.append(propagate())#output : shape user num * latdim
      self.item_propagate.append(propagate())#output : shape item num * latdim
    
    self.fc1_label = FC(2 * args.latdim, args.latdim, bias = True)
    self.fc2_label = FC(args.latdim, 1, bias = True)

    #BUG! only one <meta> is needed. 
    self.meta = meta()

  def prepareKey(self, nodeEmbed):
    key = torch.reshape(nodeEmbed @ self.K, [-1, args.att_head, args.latdim // args.att_head])
    key = torch.permute(key, (1,0,2)) #shape head_num * (user num or item num) * (latdim/head_num)
    return key
  
  def label(self, usrKey, itmKey, uHyper, iHyper):
    ulat = self.meta(uHyper, usrKey) # batchsize * latdim
    ilat = self.meta(iHyper, itmKey) # batchsize * latdim
    lat = torch.cat([ulat, ilat], dim=-1) # batchsize * 2latdim
    lat = self.fc1_label(lat, act = 'leakyrelu') #batchsize * latdim
    lat = lat + ulat + ilat
    ret = self.fc2_label(lat, act = 'sigmoid')
    ret = torch.reshape(ret, [-1]) #降维
    return ret

  def GCN(self, ulat, ilat, adj, tpadj):
    ulats = [ulat] #shape user * latdim
    ilats = [ilat] #shape item * latdim
    for i in range(args.gcn_hops):
      temulat = torch.sparse.mm(adj,ilats[-1]) #shape user * latdim  //sparse
      temilat = torch.sparse.mm(tpadj,ulats[-1]) #shape item * latdim  //sparse
      ulats.append(temulat) 
      ilats.append(temilat)
    ulats_sum = sum(ulats[1:]) #shape user * latdim
    ilats_sum = sum(ilats[1:]) #shape item * latdim
    return ulats_sum, ilats_sum

  def Regularize(self, reg, method = 'L2'):
    ret = 0.0
    for i in range(len(reg)):
      ret += torch.sum(torch.square(reg[i]))
    return ret

  def forward_test(self):
    uEmbed_gcn, iEmbed_gcn = self.GCN(self.uEmbed_ini, self.iEmbed_ini, self.adj, self.tpadj) # usre * latdim, item * latdim
    uEmbed0 = self.uEmbed_ini + uEmbed_gcn
    iEmbed0 = self.iEmbed_ini + iEmbed_gcn

    uKey = self.prepareKey(uEmbed0) #shape head_num * (user num) * (latdim/head_num)
    iKey = self.prepareKey(iEmbed0) #shape head_num * (item num) * (latdim/head_num)

    ulats = [uEmbed0]
    ilats = [iEmbed0]
    for i in range(args.gnn_layer):
      self.user_propagate[i](self.V, ulats, uKey, self.uHyper)
      self.item_propagate[i](self.V, ilats, iKey, self.iHyper)
    
    ulat = sum(ulats) #shape user * latdim
    ilat = sum(ilats) #shape item * latdim
    return ulat, ilat

  def forward(self, uid, iid, edgeids):
    #self.reg.extend([self.uEmbed_ini,self.iEmbed_ini,self.uHyper,self.iHyper])
    uEmbed_gcn, iEmbed_gcn = self.GCN(self.uEmbed_ini, self.iEmbed_ini, self.adj, self.tpadj) # usre * latdim, item * latdim
    uEmbed0 = self.uEmbed_ini + uEmbed_gcn
    iEmbed0 = self.iEmbed_ini + iEmbed_gcn

    #self.gcnNorm = (torch.sum(torch.sum(torch.square(uEmbed_gcn), dim = -1)) + torch.sum(torch.sum(torch.square(iEmbed_gcn), dim = -1))) / 2
    #self.iniNorm = (torch.sum(torch.sum(torch.square(self.uEmbed_ini), dim = -1)) + torch.sum(torch.sum(torch.square(self.iEmbed_ini), dim = -1))) / 2
    uKey = self.prepareKey(uEmbed0) #shape head_num * (user num) * (latdim/head_num)
    iKey = self.prepareKey(iEmbed0) #shape head_num * (item num) * (latdim/head_num)
    #self.reg.append(self.K)
    ulats = [uEmbed0]
    ilats = [iEmbed0]
    for i in range(args.gnn_layer):
      self.user_propagate[i](self.V, ulats, uKey, self.uHyper)
      self.item_propagate[i](self.V, ilats, iKey, self.iHyper)
    
    ulat = sum(ulats) #shape user * latdim
    ilat = sum(ilats) #shape item * latdim
    pckUlat = torch.index_select(ulat, 0, uid.int()) #shape batch size * latdim
    pckIlat = torch.index_select(ilat, 0, iid.int()) #shape batch size * latdim
    preds = torch.sum(pckUlat * pckIlat, dim = -1)

    idx = self.adj._indices() #shape (2, user * item)
    
    users, items = torch.index_select(idx[0,], 0, edgeids.int()), torch.index_select(idx[1,], 0, edgeids.int()) #shape (batchsize)
    uKey = torch.reshape(torch.permute(uKey, (1,0,2)), [-1,args.latdim]) # user num * latdim
    iKey = torch.reshape(torch.permute(iKey, (1,0,2)), [-1,args.latdim]) # item num * latdim
    userKey = torch.index_select(uKey, 0, users.int())  # batchsize * latdim
    itemKey = torch.index_select(iKey, 0, items.int())  # batchsize * latdim

    scores = self.label(userKey, itemKey, self.uHyper, self.iHyper)
    _preds = torch.sum(torch.index_select(uEmbed0, 0, users.int()) * torch.index_select(iEmbed0, 0 , items.int()), dim = -1)
    halfNum = scores.shape[0]//2
    fstScores = scores[:halfNum]
    scdScores = scores[halfNum:]
    fstPreds = _preds[:halfNum]
    scdPreds = _preds[halfNum:]

    sslLoss = torch.sum(torch.maximum(torch.Tensor([0.0]).cuda(), 1.0 - (fstPreds - scdPreds) * args.mult * (fstScores - scdScores)))
    
    reg = [self.uEmbed_ini,self.iEmbed_ini,self.uHyper,\
        self.iHyper,self.K,self.V,self.fc1_label.W_fc,\
        self.fc2_label.W_fc,self.meta.fc1.W_fc,\
        self.meta.fc2.W_fc]

    return preds, sslLoss, self.Regularize(reg, method = 'L2')