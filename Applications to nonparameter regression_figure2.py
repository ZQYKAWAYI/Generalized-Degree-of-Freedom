import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from scipy import stats


def N(n,p,se):
    X = np.random.uniform(0,1,(n,p))
    Y = np.random.randn(n)*se
    return np.mean(Y*Y),X,Y

def s(x1,x2):
    if (x1<=0.6):
        if (x2<=0.3):
            return 0
        else:
            if (x1<=0.3):
                return -2
            else:
                return -1
    else:
        if (x2<=0.8):
            return 2
        else:
            return 2.5

def T(n,p,se):
    X = np.random.uniform(0,1,(n,p))
    Y= np.zeros(n)
    noise = np.zeros(n)
    for i in range(n):
        Y[i] = s(X[i][0],X[i][1]) + np.random.randn()*se
        noise[i] = (Y[i] - s(X[i][0],X[i][1]))**2
    return np.mean(noise),X,Y

def D(T,X,Y,a,func):
    n = len(Y)
    delta_t = np.zeros(n)
    delta = np.zeros((T,n))
    u = np.zeros((T,n))
    for t in range(T):
        tmp = np.random.randn(n)*a
        delta_t = stats.norm.pdf(tmp/a,0,1)/a
        func.fit(X,Y+delta_t)
        u[t,:] = func.predict(X)
        delta[t,:] = delta_t
    ans = 0
    mean = np.mean(delta,axis=0)
    mean_u = np.mean(u,axis=0)
    for i in range(n):
        hi = np.dot(delta[:,i]-mean[i],u[:,i]-mean_u[i])/np.dot(delta[:,i]-mean[i],delta[:,i]-mean[i])
        ans += hi
    return ans
    
if __name__ == '__main__':   
    min_node = 2
    max_node = 20
    ite = 10
    plt.figure(figsize=(9,9))
    x = np.linspace(min_node,max_node,max_node-min_node+1)
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)
    plt.sca(ax1)
    plt.ylim(0,100)
    my_x_ticks = np.arange(0, 25, 5)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 110, 20)
    plt.yticks(my_y_ticks)
    plt.ylabel('GDF')
    plt.title('(a)')
    plt.sca(ax2)
    plt.ylim(0,0.6)
    my_x_ticks = np.arange(0, 25, 5)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 0.7, 0.1)
    plt.yticks(my_y_ticks)
    plt.ylabel('Est. Varience')
    plt.title('(b)')
    for i in range(ite):
        GDF = list()
        v = list()
        p = 10
        n = 100   
        se = 0.5 
        s2,X,Y = N(n,p,se)
        T1 = 100
        a = 0.5*se
        for node in range(min_node,max_node+1):
            model = tree.DecisionTreeRegressor(max_leaf_nodes=node)
            GD = D(T1,X,Y,a,model)
            GDF.append(GD)
            model.fit(X,Y)
            u = model.predict(X)
            v.append(np.dot(Y-u,Y-u)/(n-GD))
        plt.sca(ax1)
        plt.plot(x,GDF)
        plt.sca(ax2)
        plt.plot(x,v)
        
    plt.sca(ax3)
    plt.ylim(0,100)
    my_x_ticks = np.arange(0, 25, 5)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 110, 20)
    plt.yticks(my_y_ticks)
    plt.xlabel('Nodes')
    plt.ylabel('GDF')
    plt.title('(c)')
    plt.sca(ax4)
    plt.ylim(0,0.6)
    my_x_ticks = np.arange(0, 25, 5)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 0.7, 0.1)
    plt.yticks(my_y_ticks)
    plt.xlabel('Nodes')
    plt.ylabel('Est. Varience')
    plt.title('(d)')
    for i in range(ite):
        GDF = list()
        v = list()
        p = 10
        n = 100   
        se = 0.5 
        s2,X,Y = T(n,p,se)
        T1 = 100
        a = 0.5*se
        for node in range(min_node,max_node+1):
            model = tree.DecisionTreeRegressor(max_leaf_nodes=node)
            GD = D(T1,X,Y,a,model)
            GDF.append(GD)
            model.fit(X,Y)
            u = model.predict(X)
            v.append(np.dot(Y-u,Y-u)/(n-GD))
        plt.sca(ax3)
        plt.plot(x,GDF)
        plt.sca(ax4)
        plt.plot(x,v)
    