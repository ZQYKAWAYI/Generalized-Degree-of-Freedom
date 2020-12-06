import numpy as np
from scipy import stats

def D(T,X,Y,a,func):
    n = len(Y)
    delta_t = np.zeros(n)
    delta = np.zeros((T,n))
    u = np.zeros((T,n))
    for t in range(T):
        tmp = np.random.randn(n)*a
        delta_t = stats.norm.pdf(tmp/a,0,1)/a
        beta = func(X,Y+delta_t)
        u[t,:] = (X@beta).T
        delta[t,:] = delta_t
    ans = 0
    mean = np.mean(delta,axis=0)
    mean_u = np.mean(u,axis=0)
    for i in range(n):
        hi = np.dot(delta[:,i]-mean[i],u[:,i]-mean_u[i])/np.dot(delta[:,i]-mean[i],delta[:,i]-mean[i])
        ans += hi
    return ans
        
def func(X,Y):
    return np.linalg.inv(X.T@X)@X.T@Y

p = 5
n = 100   
se = 0.5
sb = 1  
GDF_list = list()
iteration = 100
for ite in range(iteration):
    noise = np.random.randn(n) * se
    beta_true = np.random.randn(p) * sb
    X = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            X[i,j]=np.random.randn(1)
    Y = X@beta_true+noise
    T = 1000
    a = 0.5*se
    ans = D(T,X,Y,a,func)
    GDF_list.append(ans)
print(np.mean(GDF_list))
