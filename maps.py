import numpy as np
#1:wall 2:start 3:goal

s1 = np.array(
    [[0,0,0,0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,1,1,1,1,1,1,1]
    ,[0,0,0,0,0,1,0,0,0,0,0,0]
    ,[0,0,0,0,0,1,0,0,0,0,0,0]
    ,[0,0,0,0,0,1,0,0,0,0,0,0]
    ,[0,0,0,0,0,1,0,0,0,0,0,0]
    ,[0,0,0,0,0,1,0,0,0,0,0,0]
    ,[0,0,0,0,0,1,0,0,0,0,0,0]
    ,[2,2,2,2,2,1,0,0,0,0,0,0]])

s0_0 = np.array(
    [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1]
    ,[0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0]
    ,[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    ,[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    ,[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    ,[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    ,[1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    ,[1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0]
    ,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    ,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    ,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    ,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    ,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    ,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    ,[1,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0]])

s2= np.array(
    [[0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,0,0,0,3]
    ,[0,0,0,0,0,0,0,0,0]
    ,[0,0,0,0,0,0,0,0,0]
    ,[0,0,0,0,0,0,0,0,0]
    ,[2,0,0,0,0,0,0,0,0]])
