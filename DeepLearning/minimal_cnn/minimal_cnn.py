import numpy as np
import struct
import pdb
import random
import cv2
import json
import argparse
import os

seed = 1234
rng = np.random.RandomState(seed)
random.seed(seed)

def my_log(x):
    x[x<1e-100] = 1e-100
    return np.log(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x.transpose()/np.sum(e_x,axis=1)).transpose()

class DataUtils(object):
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath
        
        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'    
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte
    
    def getImage(self):
        binfile = open(self._filename, 'rb')
        buf = binfile.read() 
        binfile.close()
        index = 0
        numMagic,numImgs,numRows,numCols=struct.unpack_from(self._fourBytes2,buf,index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images,dtype=np.float32)
        
    def getLabel(self):
        binFile = open(self._filename,'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, numItems= struct.unpack_from(self._twoBytes2, buf,index)
        index += struct.calcsize(self._twoBytes2)
        labels = [];
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2,buf,index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

def transform(im, trans_type):
    rows = 28
    cols = 28
    im = np.reshape(im, (rows, cols))
    if trans_type == 1:
        angle =  (np.random.rand(1) - 0.5)*30
        M = cv2.getRotationMatrix2D((rows/2,cols/2),angle,1)
        im = cv2.warpAffine(im,M,(cols,rows))
    elif trans_type == 2:
        scale = (np.random.rand(1) - 0.5)*0.1 + 1
        trans =  (np.random.rand(2) - 0.5)*8
        M = np.float32([[scale,0,trans[0]],[0,scale,trans[1]]])
        im = cv2.warpAffine(im,M,(cols,rows))
    return np.reshape(im, (1,rows*cols))

class DataProvider:
    def __init__(self, train_data, train_label, test_data, test_label, train_transform=[]):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.n_train = train_data.shape[0]
        self.n_test = test_data.shape[0]
        self.train_transform = train_transform

    def get_train_batch(self, batch_size):
        idx = np.random.randint(self.n_train, size=batch_size)
        data = np.copy(self.train_data[idx,:])
        for i in range(batch_size):
            trans_type = 0
            if len(self.train_transform) > 0:
                trans_type = np.random.randint(10086, size=1) % len(self.train_transform)
                trans_type = self.train_transform[trans_type[0]]
            if trans_type > 0:
                data[i,:] = transform(data[i,:], trans_type)
        label = np.copy(self.train_label[idx])
        return (data, label)

    def get_test_batch(self, batch_size, random=True, part=-1):
        if random:
            idx = np.random.randint(self.n_test, size=batch_size)
        else:
            idx = range(part*batch_size,(part+1)*batch_size)
        data = np.copy(self.test_data[idx,:])
        label = np.copy(self.test_label[idx])
        return (data, label)

class Layer:
    def __init__(self):
        self.learnable_data = []
        self.learnable_diff = []
    def get_learnable_data(self):
        return self.learnable_data
    def get_learnable_diff(self):
        return self.learnable_diff
    def forward(self, bottom_data):
        return None
    def backward(self, top_data, top_diff, bottom_data):
        return None
    def serialize(self):
        return {}
    def deserialize(self, dic):
        pass

class ReLU(Layer):
    def __init__(self):
        Layer.__init__(self)
    def forward(self, bottom_data):
        top_data = np.copy(bottom_data)
        top_data[top_data < 0] = 0
        return top_data
    def backward(self, top_data, top_diff, bottom_data):
        bottom_diff = np.zeros_like(top_diff)
        bottom_diff[top_data > 0] = 1
        return top_diff*bottom_diff

class Tanh(Layer):
    def __init__(self):
        Layer.__init__(self)
    def forward(self, bottom_data):
        return np.tanh(bottom_data)
    def backward(self, top_data, top_diff, bottom_data):
        tanhx = top_data
        return top_diff*(1-tanhx*tanhx)

class DropOut(Layer):
    def __init__(self, ratio):
        Layer.__init__(self)
        self.ratio = ratio
    def forward(self, bottom_data):
        self.mask = np.ones(bottom_data.shape)
        batch_size = bottom_data.shape[0]
        dim = bottom_data.shape[1]
        for i in range(batch_size):
            idx = random.sample(range(dim), int(dim*self.ratio))
            self.mask[i,idx] = 0
        top_data = self.mask*bottom_data
        return top_data
    def backward(self, top_data, top_diff, bottom_data):
        bottom_diff = self.mask*top_diff
        return bottom_diff

class Pooling(Layer):
    def __init__(self, im_size, im_channel):
        Layer.__init__(self)
        self.im_size = im_size
        self.im_channel = im_channel
    def forward(self, bottom_data):
        im_size = self.im_size
        im_channel = self.im_channel
        batch_size = bottom_data.shape[0]
        bottom_data = np.reshape(bottom_data, (batch_size,im_channel, im_size, im_size))
        top_data = np.zeros((batch_size,im_channel, im_size/2, im_size/2))
        for i in range(im_size/2):
            for j in range(im_size/2):
                # pdb.set_trace()
                val = np.max(np.max(bottom_data[:,:,i*2:i*2+2,j*2:j*2+2],axis=3),axis=2)
                top_data[:,:,i,j] = val
        top_data = np.reshape(top_data, (batch_size, im_channel*im_size/2*im_size/2))
        return top_data
    def backward(self, top_data, top_diff, bottom_data):
        im_size = self.im_size
        im_channel = self.im_channel
        batch_size = bottom_data.shape[0]
        bottom_data = np.reshape(bottom_data, (batch_size*im_channel,im_size,im_size))
        bottom_diff = np.zeros_like(bottom_data)
        top_diff = np.reshape(top_diff, (batch_size*im_channel,im_size/2,im_size/2))
        top_data = np.reshape(top_data, (batch_size*im_channel,im_size/2,im_size/2))
        for c in range(im_channel*batch_size):
            for i in range(im_size/2):
                for j in range(im_size/2):
                    diff = bottom_diff[c,i*2:i*2+2,j*2:j*2+2]
                    bd = bottom_data[c,i*2:i*2+2,j*2:j*2+2]
                    td = top_data[c,i,j]
                    idx = (bd == td)
                    tdiff = top_diff[c,i,j]
                    diff[idx] = tdiff
        return np.reshape(bottom_diff, (batch_size, im_channel*im_size*im_size))

class FC(Layer):
    def __init__(self, n_in, n_out, weight_decay=1e-5):
        Layer.__init__(self)
        high = np.sqrt(6./(n_in+n_out))
        self.W = rng.uniform(low=-high, high=high, size=(n_in, n_out))
        self.b = np.zeros((n_out))
        self.W_diff = np.zeros_like(self.W)
        self.b_diff = np.zeros_like(self.b)
        self.weight_decay = weight_decay
        self.learnable_data = [self.W, self.b]
        self.learnable_diff = [self.W_diff, self.b_diff]

    def forward(self, bottom_data):
        return np.dot(bottom_data, self.W) + self.b

    def backward(self, top_data, top_diff, bottom_data):
        self.W_diff[:] = np.dot(np.transpose(bottom_data), top_diff) + self.weight_decay*self.W
        self.b_diff[:] = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, np.transpose(self.W))
        return bottom_diff

    def serialize(self):
        return {
            'W': np.copy(self.W).tolist(), 
            'b': np.copy(self.b).tolist()}
    def deserialize(self, dic):
        self.W[:] = np.array(dic['W'])
        self.b[:] = np.array(dic['b'])

class ConvBruteforce(Layer):
    def __init__(self, channel_in, channel_out, K, size_in, size_out, weight_decay=1e-5):
        Layer.__init__(self)
        high = np.sqrt(6./(channel_in*K*K+channel_out*K*K))
        # self.k = rng.normal(0, 0.01, size=(channel_out, channel_in, K, K))
        self.k = rng.uniform(low=-high, high=high, size=(channel_out, channel_in, K, K))
        self.b = np.zeros((channel_out))
        self.k_diff = np.zeros_like(self.k)
        self.b_diff = np.zeros_like(self.b)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.K = K
        self.size_in = size_in
        self.size_out = size_out
        self.weight_decay = weight_decay
        self.learnable_data = [self.k, self.b]
        self.learnable_diff = [self.k_diff, self.b_diff]

        self.mat = np.zeros((channel_in*size_in*size_in, channel_out*size_out*size_out))
        self.mat_diff = np.zeros_like(self.mat)
        self.kernel_idx_row = [None]*(channel_out*channel_in*K*K)
        self.kernel_idx_col = [None]*(channel_out*channel_in*K*K)
        # todo: refine it
        for f in range(channel_out):
            for c in range(channel_in):
                for ii in range(size_out):
                    for jj in range(size_out):
                        # careful
                        for i in range(K):
                            for j in range(K):
                                kernel_idx = j + i*K + c*K*K + f*K*K*channel_in
                                if self.kernel_idx_row[kernel_idx] is None:
                                    self.kernel_idx_row[kernel_idx] = []
                                    self.kernel_idx_col[kernel_idx] = []
                                input_i = ii + i
                                input_j = jj + j
                                row_idx = input_j+input_i*size_in+c*size_in*size_in # pixel of input image
                                col_idx = jj + ii*size_out + f*size_out*size_out
                                self.kernel_idx_row[kernel_idx].append(row_idx)
                                self.kernel_idx_col[kernel_idx].append(col_idx)


    def forward(self, bottom_data):
        # input: channel_in x size_in x size_in
        # output: channel_out x size_out x size_out
        # batch_size = bottom_data.shape[0]
        for f in range(self.channel_out):
            for c in range(self.channel_in):
                for i in range(self.K):
                    for j in range(self.K):
                        kernel_idx = j + i*self.K + c*self.K*self.K + f*self.K*self.K*self.channel_in
                        rid = self.kernel_idx_row[kernel_idx]
                        cid = self.kernel_idx_col[kernel_idx]
                        self.mat[rid, cid] = self.k[f,c,i,j]
        b = np.zeros((bottom_data.shape[0], self.size_out*self.size_out*self.channel_out))
        for f in range(self.channel_out):
            b[:,self.size_out*self.size_out*f:self.size_out*self.size_out*(f+1)] = self.b[f]
        return np.dot(bottom_data, self.mat) + b

    def backward(self, top_data, top_diff, bottom_data):
        self.k_diff[:] = 0
        self.mat_diff[:] = np.dot(np.transpose(bottom_data), top_diff)
        for f in range(self.channel_out):
            for c in range(self.channel_in):
                for i in range(self.K):
                    for j in range(self.K):
                        kernel_idx = j + i*self.K + c*self.K*self.K + f*self.K*self.K*self.channel_in
                        rid = self.kernel_idx_row[kernel_idx]
                        cid = self.kernel_idx_col[kernel_idx]
                        self.k_diff[f,c,i,j] = np.sum(self.mat_diff[rid,cid])
        self.k_diff += self.weight_decay*self.k
        for f in range(self.channel_out):
            self.b_diff[f] = np.sum(top_diff[:,self.size_out*self.size_out*f:self.size_out*self.size_out*(f+1)])
        bottom_diff = np.dot(top_diff, np.transpose(self.mat))
        return bottom_diff

    def serialize(self):
        return {
            'k': np.copy(self.k).tolist(), 
            'b': np.copy(self.b).tolist()}
    def deserialize(self, dic):
        self.k[:] = np.array(dic['k'])
        self.b[:] = np.array(dic['b'])

def im2col(im, channel_in, size_in, K, ret, channel_out, size_out):
    # im size: channel_in*size_in*size_in, single_row
    # ret size: size_out*size_out x channel_out
    cur_row = 0
    for i_out in range(size_out):
        for j_out in range(size_out):
            idx = 0
            for c_in in range(channel_in):
                for i in range(K):
                    for j in range(K):
                        ret[cur_row, idx] = im[(j+j_out)+(i+i_out)*size_in+c_in*size_in*size_in]
                        idx += 1
            cur_row += 1

class ConvIm2Col(Layer):
    def __init__(self, channel_in, channel_out, K, size_in, size_out, weight_decay=1e-5):
        Layer.__init__(self)
        high = np.sqrt(6./(channel_in*K*K+channel_out*K*K))
        # self.k = rng.normal(0, 0.01, size=(channel_out, channel_in, K, K))
        self.k_mat = rng.uniform(low=-high, high=high, size=(K*K*channel_in, channel_out))
        self.b = np.zeros((channel_out))
        self.k_mat_diff = np.zeros_like(self.k_mat)
        self.b_diff = np.zeros_like(self.b)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.K = K
        self.size_in = size_in
        self.size_out = size_out
        self.weight_decay = weight_decay
        self.learnable_data = [self.k_mat, self.b]
        self.learnable_diff = [self.k_mat_diff, self.b_diff]

        self.im_col_mat = np.zeros((size_out*size_out, K*K*channel_in))
        self.conv_res_mat = np.zeros((size_out*size_out, channel_out))
        self.aux_row = [None]*(size_in*size_in*channel_in)
        self.aux_col = [None]*(size_in*size_in*channel_in)
        cur_row = 0
        for i_out in range(size_out):
            for j_out in range(size_out):
                idx = 0
                for c_in in range(channel_in):
                    for i in range(K):
                        for j in range(K):
                            im_idx = (j+j_out)+(i+i_out)*size_in+c_in*size_in*size_in
                            if self.aux_row[im_idx] is None:
                                self.aux_row[im_idx] = []
                                self.aux_col[im_idx] = []
                            row_idx = cur_row
                            col_idx = idx
                            self.aux_row[im_idx].append(row_idx)
                            self.aux_col[im_idx].append(col_idx)
                            idx += 1
                cur_row += 1

    def forward(self, bottom_data):
        # input: channel_in x size_in x size_in
        # output: channel_out x size_out x size_out
        # im2col:
        n = bottom_data.shape[0]
        ret = np.zeros((n, self.size_out*self.size_out*self.channel_out))
        for i in range(n):
            # im2col(im, channel_in, size_in, K, ret, channel_out, size_out)
            im = bottom_data[i,:]
            im2col(im, self.channel_in, self.size_in, self.K, self.im_col_mat, self.channel_out, self.size_out)
            self.conv_res_mat = self.im_col_mat.dot(self.k_mat)
            for f in range(self.channel_out):
                self.conv_res_mat[:,f] += self.b[f]
            ret[i, :] = self.conv_res_mat.flatten()
        return ret

    def backward(self, top_data, top_diff, bottom_data):
        self.k_mat_diff[:] = 0
        bottom_diff = np.zeros_like(bottom_data)
        n = top_diff.shape[0]
        for i in range(n):
            diff_row = top_diff[i,:]
            diff_mat = np.resize(diff_row, (self.size_out*self.size_out, self.channel_out))
            im = bottom_data[i,:]
            im2col(im, self.channel_in, self.size_in, self.K, self.im_col_mat, self.channel_out, self.size_out)
            self.k_mat_diff[:] += np.transpose(self.im_col_mat).dot(diff_mat)
            bottom_diff_mat = diff_mat.dot(np.transpose(self.k_mat))
            for c in range(self.channel_in):
                for ii in range(self.size_in):
                    for jj in range(self.size_in):
                        im_idx = jj+ii*self.size_in+c*self.size_in*self.size_in
                        rid = self.aux_row[im_idx]
                        cid = self.aux_col[im_idx]
                        bottom_diff[i,im_idx] = np.sum(bottom_diff_mat[rid, cid])
        self.k_mat_diff += self.weight_decay*self.k_mat
        for f in range(self.channel_out):
            self.b_diff[f] = np.sum(top_diff[:,self.size_out*self.size_out*f:self.size_out*self.size_out*(f+1)])
        return bottom_diff

    def serialize(self):
        return {
            'k': np.copy(self.k_mat).tolist(), 
            'b': np.copy(self.b).tolist()}
    def deserialize(self, dic):
        self.k_mat[:] = np.array(dic['k'])
        self.b[:] = np.array(dic['b'])

Conv = ConvBruteforce
# Conv = ConvIm2Col

class SoftmaxWithLoss:
    def __init__(self):
        pass
    def forward(self, bottom_data, label):
        # each row as a sample, bottom_data -> prob
        # bottom_data: [n_batch, n_outputchannel]
        # label: [n_batch]
        n = bottom_data.shape[0]
        probs = softmax(bottom_data)
        pred_label = np.argmax(bottom_data, axis=1)
        loss = np.mean(-my_log(probs[np.array(range(n)),label]))
        diff = np.copy(probs)
        diff[np.array(range(n)),label] -= 1
        diff /= n
        acc = np.mean(pred_label==label)
        return loss,diff,acc

class SGD:
    def __init__(self, momentum, learnable_data, learnable_diff):
        self.learnable_data = learnable_data
        self.learnable_diff = learnable_diff
        self.last_diff = [0]*len(learnable_diff)
        self.momentum = momentum
    def step(self, lr):
        n = len(self.learnable_diff)
        for i in range(n):
            self.last_diff[i] = self.last_diff[i]*self.momentum + self.learnable_diff[i]*(1 - self.momentum)
            self.learnable_data[i] -= self.last_diff[i]*lr

class Net:
    def __init__(self):
        self.layers = []
        self.layer_output_data = []    # tht output of the layer
        self.input_data = None
        self.optim = None

    def add_layer(self, layer):
        self.layers.append(layer)
        self.layer_output_data.append(None)

    def get_learnable_data(self):
        learnable_data = []
        for layer in self.layers:
            learnable_data = learnable_data + layer.get_learnable_data()
        return learnable_data
    def get_learnable_diff(self):
        learnable_diff = []
        for layer in self.layers:
            learnable_diff = learnable_diff + layer.get_learnable_diff()
        return learnable_diff

    def forward(self, input_data):
        self.input_data = input_data
        n = len(self.layers)
        for i in range(n):
            output_data = self.layers[i].forward(input_data)
            self.layer_output_data[i] = output_data
            input_data = output_data
        return output_data

    def backward(self, top_diff):
        n = len(self.layers)
        for ii in range(n):
            i = n - ii - 1
            top_data = self.layer_output_data[i]
            bottom_data = self.layer_output_data[i-1] if i > 0 else self.input_data
            bottom_diff = self.layers[i].backward(top_data, top_diff, bottom_data)
            top_diff = bottom_diff

    def update(self, lr):
        if self.optim is None:
            self.optim = SGD(0.9, self.get_learnable_data(), self.get_learnable_diff())
        self.optim.step(lr)

    def save(self, model_path):
        n = len(self.layers)
        json_data = {}
        for i in range(n):
            item = self.layers[i].serialize()
            layer_name = str(i)
            json_data[layer_name] = item
        with open(model_path, 'w') as f:
            json.dump(json_data, f)
            print 'saved model to', model_path

    def load(self, model_path):
        with open(model_path, 'r') as f:
            json_data = json.load(f)
            n = len(self.layers)
            assert(n == len(json_data))
            for i in range(n):
                layer_name = str(i)
                self.layers[i].deserialize(json_data[layer_name])
            print 'loaded model from', model_path

def train(data_provider, net, n_iter, batch_size, lr):
    loss_layer = SoftmaxWithLoss()
    for i in range(n_iter):
        (bottom_data,label) = data_provider.get_train_batch(batch_size)
        fc_out = net.forward(bottom_data)
        loss,top_diff,acc = loss_layer.forward(fc_out, label)
        net.backward(top_diff)
        net.update(lr)

        (bottom_data,label) = data_provider.get_test_batch(batch_size)
        fc_out = net.forward(bottom_data)
        test_loss,_,test_acc = loss_layer.forward(fc_out, label)
        print 'iter: %d, train: loss: %.4f, acc: %.4f; test: loss:%.4f, acc: %.4f' % (i,loss,acc, test_loss,test_acc)

def test(data_provider, net):
    test_accs = []
    loss_layer = SoftmaxWithLoss()
    batch_size = 100
    n = data_provider.test_label.shape[0]/batch_size
    for i in range(n):
        (bottom_data,label) = data_provider.get_test_batch(batch_size, False, i)
        fc_out = net.forward(bottom_data)
        test_loss,_,test_acc = loss_layer.forward(fc_out, label)
        test_accs.append(test_acc)
    return np.mean(np.array(test_accs))

def main(args):
    # load data
    data_dir = args.mnist_dir
    train_data = DataUtils(os.path.join(data_dir, 'train-images.idx3-ubyte')).getImage()
    train_label = DataUtils(os.path.join(data_dir, 'train-labels.idx1-ubyte')).getLabel()
    test_data = DataUtils(os.path.join(data_dir, 't10k-images.idx3-ubyte')).getImage()
    test_label = DataUtils(os.path.join(data_dir, 't10k-labels.idx1-ubyte')).getLabel()
    trans_type = [0,1,2]
    data_provider = DataProvider(train_data, train_label, test_data, test_label, trans_type)

    net = Net()
    if args.net_type == 'mlpv1':
        ######## MLP_V1 ########
        net.add_layer(FC(28*28, 500))
        net.add_layer(ReLU())
        net.add_layer(DropOut(0.5))
        net.add_layer(FC(500, 800))
        net.add_layer(ReLU())
        net.add_layer(DropOut(0.5))
        net.add_layer(FC(800, 300))
        net.add_layer(ReLU())
        net.add_layer(DropOut(0.5))
        net.add_layer(FC(300, 10))
    elif args.net_type == 'mlpv2':
        ######## MLP_V2 ########
        net.add_layer(FC(28*28, 1200))
        net.add_layer(Tanh())
        net.add_layer(DropOut(0.8))
        net.add_layer(FC(1200, 10))
    elif args.net_type == 'conv':
        ######## CONVNET ########
        # too slow, think over before use it 
        net.add_layer(Conv(1,32,5,28,24))
        net.add_layer(Pooling(24,32))
        net.add_layer(ReLU())
        net.add_layer(DropOut(0.5))
        net.add_layer(Conv(32,64,5,12,8))
        net.add_layer(Pooling(8,64))
        net.add_layer(ReLU())
        net.add_layer(DropOut(0.5))
        net.add_layer(FC(64*4*4, 500))
        net.add_layer(ReLU())
        net.add_layer(DropOut(0.5))
        net.add_layer(FC(500, 10))

    model_path = args.model_path
    if os.path.isfile(model_path):
        net.load(model_path)
    if args.phase & 1 > 0:
        lr = args.lr
        for i in range(args.step):
            train(data_provider, net, args.iter, args.batch_size, lr)
            lr *= args.lr_scale
        net.save(model_path)
    # train(data_provider, net, 20000, batch_size, 1)
    # train(data_provider, net, 20000, batch_size, 0.1)
    # train(data_provider, net, 20000, batch_size, 0.01)
    # train(data_provider, net, 20000, batch_size, 0.001)
    # train(data_provider, net, 20000, batch_size, 0.0001)
    # train(data_provider, net, 20000, batch_size, 0.00001)
    # train(data_provider, net, 20000, batch_size, 0.000001)
    if args.phase & 2 > 0:
        test_acc = test(data_provider, net)
        print 'final test accuracy:', test_acc

        # data_provider = DataProvider(train_data, train_label, train_data, train_label, trans_type)
        data_provider.test_data = data_provider.train_data
        data_provider.test_label = data_provider.train_label
        train_acc = test(data_provider, net)
        print 'final train accuracy:', train_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', help='type of network, try mlpv1, mlpv2 or conv', default='mlpv1')
    parser.add_argument('--mnist_dir', help='directory of mnist data', default='.')
    parser.add_argument('--phase', type=int, help='1:train, 2:test, 3:train and test', default=1)
    parser.add_argument('--model_path', help='path to save or load model', default='mlp.json')
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=128)
    parser.add_argument('--iter', type=int, help='number of iter for each learning rate in training', default=20000)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=1)
    parser.add_argument('--step', type=int, help='steps to scale down learning rate', default=7)
    parser.add_argument('--lr_scale', type=float, help='scale down learning rate by this ratio after iter iterations', default=0.1)
    args = parser.parse_args()
    main(args)
    