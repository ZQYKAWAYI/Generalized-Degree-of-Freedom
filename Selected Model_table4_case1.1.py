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
    if (K>=4 and K<=(q-4)):
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
    else:
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
    # case 1.1
    rou = 0.5
    n = 22
    q = 20
    alpha = 0
    se = 1
    beta = np.zeros(q)
    simulation = 100
    #True Loss
    EAIC_selected_list = list()
    AIC_selected_list = list()
    GCV_df_selected_list = list()
    GCV_gdf_selected_list = list()
    GCV2_df_selected_list = list()
    GCV2_gdf_selected_list = list() 
    #Average number of variables selected
    EAIC_V_list = list()
    AIC_V_list = list()
    GCV_df_V_list = list()
    GCV_gdf_V_list = list()
    GCV2_df_V_list = list()
    GCV2_gdf_V_list = list()
    #s^2(adj)
    GCV_df_s_adj_list = list()
    GCV2_df_s_adj_list = list()
    #s^2(cor)
    GCV_df_s_cor_list = list()
    GCV2_df_s_cor_list = list()
    GCV_gdf_s_cor_list = list()
    GCV2_gdf_s_cor_list = list()
    K_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #K_list = [0,1,2,3]
    for sim in range(simulation):
        X,Y = data(alpha,n,beta,se,rou)
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
        GCV_gdf_list = list()
        GCV2_gdf_list = list()
        GCV_df_list = list()
        GCV2_df_list = list()
        print(sim+1)
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
            GCV_df = np.dot(Y-u_hat,Y-u_hat)/((n-(K+1))*(n-(K+1)))
            GCV_gdf = np.dot(Y-u_hat,Y-u_hat)/((n-GDF)*(n-GDF))
            #print('\t',K)
            '''
            print('GDF=',GDF)
            print('AIC=',AIC)
            print('Loss=',Loss)
            print('EAIC=',EAIC)
            print('s^2_adj=',s_adj)
            print('s^2_cor=',s_cor)
            print('R^2_adj=',R_adj)
            print('R^2_cor=',R_cor)
            print('\n')
            '''
            GDF_list.append(GDF)
            AIC_list.append(AIC)
            Loss_list.append(Loss)
            EAIC_list.append(EAIC)
            s_adj_list.append(s_adj)
            s_cor_list.append(s_cor)
            R_adj_list.append(R_adj)
            R_cor_list.append(R_cor)
            GCV_gdf_list.append(GCV_gdf)
            GCV_df_list.append(GCV_df)
            if (K<=15):
                GCV2_df_list.append(GCV_df)
                GCV2_gdf_list.append(GCV_gdf)
        EAIC_selected = np.argmin(EAIC_list)
        AIC_selected = np.argmin(AIC_list)
        GCV_df_selected = np.argmin(GCV_df_list)
        GCV_gdf_selected = np.argmin(GCV_gdf_list)
        GCV2_df_selected = np.argmin(GCV2_df_list)
        GCV2_gdf_selected = np.argmin(GCV2_gdf_list)
        #True Loss
        EAIC_selected_list.append(Loss_list[EAIC_selected])
        AIC_selected_list.append(Loss_list[AIC_selected])
        GCV_df_selected_list.append(Loss_list[GCV_df_selected])
        GCV_gdf_selected_list.append(Loss_list[GCV_gdf_selected])
        GCV2_df_selected_list.append(Loss_list[GCV2_df_selected])
        GCV2_gdf_selected_list.append(Loss_list[GCV2_gdf_selected])        
        #Average number of variables selected
        EAIC_V_list.append(EAIC_selected)
        AIC_V_list.append(AIC_selected)
        GCV_df_V_list.append(GCV_df_selected)
        GCV_gdf_V_list.append(GCV_gdf_selected)
        GCV2_df_V_list.append(GCV2_df_selected)
        GCV2_gdf_V_list.append(GCV2_gdf_selected)
        #s^2(adj)
        GCV_df_s_adj_list.append(s_adj_list[GCV_df_selected])
        GCV2_df_s_adj_list.append(s_adj_list[GCV2_df_selected])
        #s^2(cor)
        GCV_df_s_cor_list.append(s_cor_list[GCV_df_selected])
        GCV_gdf_s_cor_list.append(s_cor_list[GCV_gdf_selected])
        GCV2_df_s_cor_list.append(s_cor_list[GCV2_df_selected])
        GCV2_gdf_s_cor_list.append(s_cor_list[GCV2_gdf_selected])
        
print('\tbeta\t\t0_20')
print('\t\t\tTrue Loss')
print("\tEAIC\t\t%.2f" % np.mean(EAIC_selected_list))
print("\tAIC\t\t%.2f" % np.mean(AIC_selected_list))
print("\tGCV_gdf\t\t%.2f" % np.mean(GCV_gdf_selected_list))
print("\tGCV^*_gdf\t%.2f" % np.mean(GCV2_gdf_selected_list))
print("\tGCV_df\t\t%.2f" % np.mean(GCV_df_selected_list))
print("\tGCV^*_df\t%.2f" % np.mean(GCV2_df_selected_list))
print('\t\tAverage number of variables selected')
print("\tEAIC\t\t%.2f" % np.mean(EAIC_V_list))
print("\tAIC\t\t%.2f" % np.mean(AIC_V_list))
print("\tGCV_gdf\t\t%.2f" % np.mean(GCV_gdf_V_list))
print("\tGCV^*_gdf\t%.2f" % np.mean(GCV2_gdf_V_list))
print("\tGCV_df\t\t%.2f" % np.mean(GCV_df_V_list))
print("\tGCV^*_df\t%.2f" % np.mean(GCV2_df_V_list))
print('\t\t\ts^2(adj)')
print("\tGCV_df\t\t%.2f" % np.mean(GCV_df_s_adj_list))
print("\tGCV^*_df\t%.2f" % np.mean(GCV2_df_s_adj_list))
print('\t\t\ts^2(cor)')
print("\tGCV_gdf\t\t%.2f" % np.mean(GCV_gdf_s_cor_list))
print("\tGCV^*_gdf\t%.2f" % np.mean(GCV2_gdf_s_cor_list))
print("\tGCV_df\t\t%.2f" % np.mean(GCV_df_s_cor_list))
print("\tGCV^*_df\t%.2f" % np.mean(GCV2_df_s_cor_list))
