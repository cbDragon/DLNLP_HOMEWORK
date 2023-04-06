import math
import numpy as np
import csv

def GaussProb(x,miu,sig):
    y = 1/(math.sqrt(2*math.pi)*sig) * math.exp(-(x-miu)**2/2/sig**2)
    return y

if __name__ == "__main__":
    file_locaton = 'C:\\Users\\Long\\Desktop\\lxj\\course_graduate\\DLNLP\\work2\\height_data.csv'
    
    with open(file_locaton, "r") as f:
        reader = csv.DictReader(f)
        tempt_h = [row['height'] for row in reader]
    
    #Read height to data_h
    tempt_h = np.array(tempt_h)
    N = np.size(tempt_h, 0)
    data_h = np.arange(N, dtype=float)
    for i in range(N):
        data_h[i] = float(tempt_h[i])
    K =  2

    #Initial parameters
    theta = np.array([[170.,5],[170,5]])
    alpha = np.array([0.5, 0.5])
    epi = 1e-4
    print(theta)
    L = 0
    count = 0 
    gama = np.arange(N*K,dtype = float).reshape(K,N)
    while(count < 100):
        #update gama
        for k in range(K):
            for j in range(N):
                tempt = 0
                for p in range(K):
                    tempt += alpha[p]*GaussProb(data_h[j], theta[p][0], math.sqrt(theta[p][1]))
                gama[k][j] = alpha[k]*GaussProb(data_h[j], theta[k][0], math.sqrt(theta[k][1])) / tempt
        
        #update miu
        for k in range(K):
            tempt1 = 0
            tempt2 = 0
            for j in range(N):
                tempt1 += gama[k][j] * data_h[j]
                tempt2 += gama[k][j]
            theta[k][0] = tempt1 / tempt2

        #update sig
        for k in range(K):
            tempt1 = 0
            tempt2 = 0
            for j in range(N):
                tempt1 += gama[k][j] * (data_h[j] - theta[k][0])**2
                tempt2 += gama[k][j]
            theta[k][1] = tempt1 / tempt2

        #update alpha
        for k in range(K):
            tempt1 = 0
            for j in range(N):
                tempt1 += gama[k][j]
            alpha[k] = tempt1 / N

        count = count + 1
    theta[0][1] = np.sqrt(theta[0][1])
    theta[1][1] = np.sqrt(theta[1][1])
    print(theta)

    
