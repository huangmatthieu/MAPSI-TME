import numpy as np
import matplotlib.pyplot as plt
plt.close('all')      


fname = "dataVelib.pkl"
f= open(fname,'rb')
data = pkl.load(f)
f.close()
m0 = np.array([[1, 2], [3, 4]]) 
print m0
m1 = np.ones((3,5))
print m1
v2 = np.linspace(0, 10, 15)
print v2

A = np.random.randn(5,6)
print A
print A[::]

B = np.array([1,2,3,4,5])
print B[1:3] 

m6 = np.vstack((np.array([[1, 2], [3, 4]]), np.ones((3,3))))
print m6