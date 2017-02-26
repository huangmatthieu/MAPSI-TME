import numpy as np
import matplotlib.pyplot as plt
plt.close('all') 

v=np.array([np.arange(1,11)])
v2=np.random.rand(1,10)
v3=np.zeros((10,1))
v4=np.array([np.arange(1,4)])
m1=np.vstack((v4,np.hstack((v.T,v2.T,v3))))
print m1