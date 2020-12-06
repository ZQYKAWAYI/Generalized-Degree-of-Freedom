import numpy as np
import itertools
from scipy import stats

def data(alpha,n,beta,se,rou):
    cov = (1-rou)*np.identity(q) + rou*(np.ones((q,q)))
    X = np.random.multivariate_normal(np.zeros(q), cov, n)
    Y = alpha + X@beta + np.random.randn(n)*se
    return X,Y

def Y_m(X,Y,*args):
    q = len(X[0])
    n = len(Y)
    K = args[0]
    line = np.arange(0,q,1)
    min_error = np.inf
    Ym = np.zeros(n)
    '''
    xk = np.zeros((n,K))
    for i in range(K):
        xk[:,i] = X[:,i]
    Y_hat = xk@np.linalg.inv(xk.T@xk)@xk.T@Y
    min_error = np.dot(Y-Y_hat,Y-Y_hat)
    Ym = Y_hat
    for i in range(1000):
        xk = np.zeros((n,K))
        index = np.sort(np.random.choice(q, K, replace=False))
        for j in range(K):
            xk[:,j] = X[:,index[j]]
        Y_hat = xk@np.linalg.inv(xk.T@xk)@xk.T@Y
        error = np.dot(Y-Y_hat,Y-Y_hat)
        if (error<min_error):
            Ym = Y_hat
            min_error = error        
    '''
    for i in itertools.combinations(line,K):
        xk = np.zeros((n,K))
        for j,ind in enumerate(i):
            xk[:,j] = X[:,ind]
        Y_hat = xk@np.linalg.inv(xk.T@xk)@xk.T@Y
        error = np.dot(Y-Y_hat,Y-Y_hat)
        if (error<min_error):
            Ym = Y_hat
            min_error = error
    return Ym

def D(T,X,Y,a,func,*args):
    n = len(Y)
    delta_t = np.zeros(n)
    delta = np.zeros((T,n))
    u = np.zeros((T,n))
    for t in range(T):
        tmp = np.random.randn(n)
        delta_t = stats.norm.pdf(tmp/a,0,1)/a
        u[t,:] = func(X,Y+delta_t,*args)
        delta[t,:] = delta_t
    ans = 0
    mean = np.mean(delta,axis=0)
    mean_u = np.mean(u,axis=0)
    for i in range(n):
        hi = np.dot(delta[:,i]-mean[i],u[:,i]-mean_u[i])/np.dot(delta[:,i]-mean[i],delta[:,i]-mean[i])
        ans += hi
    return ans
    
if __name__=="__main__":
    rou = 0.5
    n = 22
    q = 20
    alpha = 0
    se = 1
    # case 2
    beta = np.zeros(q)
    beta[0] = beta[1] = beta[2] = beta[3] = beta[4] = 2
    #X,Y = data(alpha,n,beta,se,rou)
    X = np.load('X_case2.npy')
    Y = np.load('Y_case2.npy')
    T = 100
    a = 0.5*se
    GDF_list = list()
    AIC_list = list()
    Loss_list = list()
    EAIC_list = list()
    s_adj_list = list()
    s_cor_list = list()
    R_adj_list = list()
    R_cor_list = list()
    K_list = [1,5,6,10,15,20]
    for K in K_list:
        GDF = D(T,X,Y,a,Y_m,K)
        u_hat = Y_m(X,Y,K)
        Loss = np.dot(alpha+X@beta-u_hat,alpha+X@beta-u_hat) 
        AIC = np.dot(Y-u_hat,Y-u_hat)-n*se*se+2*(K+1)*se*se
        s_adj = np.dot(Y-u_hat,Y-u_hat)/(n-(K+1))   
        R_adj = 1-(s_adj)/(Y.T@Y/n)
        EAIC = np.dot(Y-u_hat,Y-u_hat)-n*se*se+2*GDF*se*se
        s_cor =  np.dot(Y-u_hat,Y-u_hat)/(n-GDF)
        R_cor = 1-(s_cor)/(Y.T@Y/n)
        print(K)
        print('GDF=',GDF)
        print('AIC=',AIC)
        print('Loss=',Loss)
        print('EAIC=',EAIC)
        print('s^2_adj=',s_adj)
        print('s^2_cor=',s_cor)
        print('R^2_adj=',R_adj)
        print('R^2_cor=',R_cor)
        print('\n')
        GDF_list.append(GDF)
        AIC_list.append(AIC)
        Loss_list.append(Loss)
        EAIC_list.append(EAIC)
        s_adj_list.append(s_adj)
        s_cor_list.append(s_cor)
        R_adj_list.append(R_adj)
        R_cor_list.append(R_cor)
    
    print('\tK\t',end='')
    for i in range(len(K_list)):
        print("\t",K_list[i],end='')
    print('\n',end='')
    print('\tGDF\t',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % GDF_list[i],end='')
    print('\n',end='')
    print('\tAIC\t',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % AIC_list[i],end='')
    print('\n',end='')
    print('\tLoss\t',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % Loss_list[i],end='')
    print('\n',end='')    
    print('\tEAIC\t',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % EAIC_list[i],end='')
    print('\n',end='')
    print('\ts^2(adj)',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % s_adj_list[i],end='')
    print('\n',end='')    
    print('\ts^2(cor)',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % s_cor_list[i],end='')
    print('\n',end='') 
    print('\tR^2(adj)',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % R_adj_list[i],end='')
    print('\n',end='') 
    print('\tR^2(cor)',end='')
    for i in range(len(K_list)):
        print("\t%.2f" % R_cor_list[i],end='')
    print('\n',end='') 

 