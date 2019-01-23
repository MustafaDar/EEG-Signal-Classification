import csv
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd 

import time


class ProcessData:
        
        def __init__(self,profilepath,profilename):
                self.profile_name = profilename
                self.profile_path = profilepath
                self.features = []
                self.labels = []
                
        def do_fft(self,all_channel_data): 
                """
                Do fft in each channel for all channels.
                Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
                Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
                """
                data_fft = map(lambda x: np.fft.fft(x),all_channel_data)
                
                return data_fft

        def get_frequency(self,all_channel_data): 
                """
                Get frequency from computed fft for all channels. 
                Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
                Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
                """
                #Length data channel
                L = len(all_channel_data[0])

                #Sampling frequency
                Fs = 128

                #Get fft data
                data_fft = self.do_fft(all_channel_data)

                #Compute frequency
                frequency = map(lambda x: abs(x/L),data_fft)
                frequency = map(lambda x: x[: L/2+1]*2,frequency)

                #List frequency
                delta = map(lambda x: x[L*1/Fs-1: L*4/Fs],frequency)
                theta = map(lambda x: x[L*4/Fs-1: L*8/Fs],frequency)
                alpha = map(lambda x: x[L*5/Fs-1: L*13/Fs],frequency)
                beta = map(lambda x: x[L*13/Fs-1: L*30/Fs],frequency)
                gamma = map(lambda x: x[L*30/Fs-1: L*50/Fs],frequency)
                print len(theta[0])

                return delta,theta,alpha,beta,gamma


        def get_feature(self,all_channel_data): 
                #Get frequency data
                (delta,theta,alpha,beta,gamma) = self.get_frequency(all_channel_data)

                #Compute feature std
                delta_std = np.std(delta, axis=1)
                theta_std = np.std(theta, axis=1)
                alpha_std = np.std(alpha, axis=1)
                beta_std = np.std(beta, axis=1)
                gamma_std = np.std(gamma, axis=1)

                #Compute feature mean
                delta_m = np.mean(delta, axis=1)
                theta_m = np.mean(theta, axis=1)
                alpha_m = np.mean(alpha, axis=1)
                beta_m = np.mean(beta, axis=1)
                gamma_m = np.mean(gamma, axis=1)

                #Concate feature
                feature = np.array([delta_std,delta_m,theta_std,theta_m,alpha_std,alpha_m,beta_std,beta_m,gamma_std,gamma_m])
                feature = feature.T
                feature = feature.ravel()
                
                return feature
        
        def grid_searchcv(self,X,y):
                print(str(len(X))+'==>'+str(len(y)))

                index_of_A = [i for i in range(len(y)) if y[i] == 1]
                index_of_B = [i for i in range(len(y)) if y[i] == 2]

                self.X = X
                self.y = y
                
                A = len(y[y==1])
                B = len(y[y==2])
                C = 0
                
                if A>B:
                        print('Right have minimum number of trials=>',B)
                        C = B
                else:
                        print('Left have minimum number of trials=>',A)
                        C = A
                i=0
                for indexA in index_of_A:
                        X[i]=X[indexA]
                        y[i] = 1
                        i = i+1
                        if i==C:
                                break
                for indexB in index_of_B:
                        X[i]=X[indexB]
                        y[i]=2
                        i = i+1
                        if i==(C*2):
                                break
                
                ly = len(y)

                for k in range(i,ly):
                        X = np.delete(X, (k),axis=0)
                        y = np.delete(y, (k),axis=0)

                print('x==>'+str(len(X)),' y==>'+str(len(y)))

                for_test = (20*(C*2))/100   #e.g 8
                #for_test=8
                
                np.random.seed(0)
                indices = np.random.permutation(len(X))
                X_train = X[indices[:-for_test]] 
                y_train = y[indices[:-for_test]]
                X_final_test = X[indices[-for_test:]]
                y_final_test = y[indices[-for_test:]]

                X = X_train
                y = y_train
                
                min_c = -5
                max_c = 15
                C_range = [2**i for i in range(min_c,max_c+1)]

                min_gamma = -10
                max_gamma = 5
                gamma_range = [2**i for i in range(min_gamma,max_gamma+1)]

                print("# Tuning hyper-parameters")

                param_grid = dict(gamma=gamma_range, C=C_range)
                cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)  #####see page 1795 for doc
                grid.fit(X, y)
                
                print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
                clf = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
                clf.fit(X,y)
                print(np.shape(X_final_test))
                y_predict = clf.predict(X_final_test)
                print('origianl',y_final_test,' predicted ',y_predict)
                joblib.dump(clf, os.path.join(self.profile_path,self.profile_name+'_model.pkl'))
                #return grid.best_params_[0],grid.best_params_[1],grid.best_score_

        def main_process(self):

                trials_data_rows = np.zeros((1,17921)) #auto change
                
                with file(self.profile_path+self.profile_name+'.csv') as f:
                        trial_data_rows = np.loadtxt(f,delimiter=',')

                        ref_features = np.zeros((1,140),dtype='float64')
                        actions_features = np.zeros((1,140),dtype='float64')
                        lbls = []
                        for row_num in range(0,len(trial_data_rows)):  #len(trial_data_rows)
                                sensors_data = np.zeros((14,1280))
                                for t in range(0,10):
                                        for sensor in range(0,14):
                                                data = np.zeros((1,128))
                                                data[0,0:127] = trial_data_rows[row_num,((sensor*128)+(t*1792)):((sensor*128)+(t*1792)+127)]
                                                sensors_data[sensor,(128*t):((128*t)+127)] = data[0,0:127]
                                #print(sensors_data)
                                
                                ref_features[0] = self.get_feature(sensors_data[:,0:639])
                                actions_features[0] = self.get_feature(sensors_data[:,640:1279])
                                
                                self.features.append(np.absolute((ref_features[0]-actions_features[0])/ref_features[0]))
                                lbls.append(trial_data_rows[row_num,-1:])
    
                        self.labels = [int(lbls[i][0]) for i in range(0,len(lbls))]
                        
                        df = pd.DataFrame(self.features)
                        df[140] = self.labels
                        df.to_csv(self.profile_path + self.profile_name+'_Features.csv', header=None)

                                
                self.grid_searchcv(np.array(self.features),np.array(self.labels))
                #print(C,gamma,score)

