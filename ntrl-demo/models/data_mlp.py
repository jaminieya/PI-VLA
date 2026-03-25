import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad
#import open3d as o3d

class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, points, speed, normal):
        # Creating identical pairs
        points    = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        normal  = Variable(Tensor(normal))
        self.data=torch.cat((points,speed,normal),dim=1)
        #self.grid  = Variable(Tensor(grid))

    def send_device(self,device):
        self.data    = self.data.to(device)

    def __getitem__(self, index):
        data = self.data[index]
        #print(index)
        return data, index
    def __len__(self):
        return self.data.shape[0]

class _TrajectoryDataset(torch.utils.data.Dataset):
    """Dataset for trajectory-based NTField training: (points, tau_obs) only."""
    def __init__(self, points, tau_obs):
        points = Variable(Tensor(points))
        tau_obs = Variable(Tensor(tau_obs)).unsqueeze(1)
        self.data = torch.cat((points, tau_obs), dim=1)

    def send_device(self, device):
        self.data = self.data.to(device)

    def __getitem__(self, index):
        data = self.data[index]
        return data, index

    def __len__(self):
        return self.data.shape[0]


def DatabaseTrajectory(PATH):
    """
    Load trajectory dataset for NTField training.
    Expects {PATH}/points.npy (N, 12) and {PATH}/tau_obs.npy (N,).
    """
    points = np.load('{}/points.npy'.format(PATH))
    tau_obs = np.load('{}/tau_obs.npy'.format(PATH))
    print(points.shape)
    print(tau_obs.shape)
    assert points.shape[0] == tau_obs.shape[0]
    database = _TrajectoryDataset(points, tau_obs)
    return database


def Database(PATH):
    
    #try:
    points = np.load('{}/sampled_points.npy'.format(PATH))#[:100000,:]
    speed = np.load('{}/speed.npy'.format(PATH))#[:100000,:]
    normal = np.load('{}/normal.npy'.format(PATH))#[:100000,:]
    #occupancies = np.unpackbits(np.load('{}/voxelized_point_cloud_128res_20000points.npz'.format(PATH))['compressed_occupancies'])
    #input = np.reshape(occupancies, (128,)*3)
    #grid = np.array(input, dtype=np.float32)
    #print(tau.min())
    #p0 = np.random.rand(100000,2)-0.5
    #s0 = sdf(p0)
    #p1 = np.random.rand(100000,2)-0.5
    #s1 = sdf(p1)
    #points = np.concatinate((p0,p1),axis=1)
    #speed = np.concatinate((s0,s1),axis=1)
    print(points.shape)
    print(speed.shape)
    print(normal.shape)
        #points[:,2:]=0
        #speed[:,1:]=1
    #except ValueError:
    #    print('Please specify a correct source path, or create a dataset')
    rows=points.shape[0]


    print(points.shape,speed.shape)
    #print(np.shape(grid))
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed,normal)
    #database = _numpy2dataset(XP,YP)
    return database





