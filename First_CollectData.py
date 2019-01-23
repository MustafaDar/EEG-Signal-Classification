import os
import ctypes
import sys
from ctypes import *
from numpy import *
import numpy as np
import time
import random
from ctypes.util import find_library
print ctypes.util.find_library('edk.dll')  
print os.path.exists('.\\edk.dll')
libEDK = cdll.LoadLibrary(".\\edk.dll")

class CollectData:
    def __init__(self,self_kivy,profileloc,profilename):  #(self,location,profile_name):
        
        self.trial_data_row = np.zeros((1,17921))
        self.choice = ''
        self.time = 0
        self.profile_loc = profileloc
        self.profile_name = profilename
        
        self.self_kivy = self_kivy
        self.exit = 0 #kivy_exit
        
        self.ED_COUNTER = 0
        self.ED_INTERPOLATED=1
        self.ED_RAW_CQ=2
        self.ED_AF3=3
        self.ED_F7=4
        self.ED_F3=5
        self.ED_FC5=6
        self.ED_T7=7
        self.ED_P7=8
        self.ED_O1=9
        self.ED_O2=10
        self.ED_P8=11
        self.ED_T8=12
        self.ED_FC6=13
        self.ED_F4=14
        self.ED_F8=15
        self.ED_AF4=16
        self.ED_GYROX=17
        self.ED_GYROY=18
        self.ED_TIMESTAMP=19
        self.ED_ES_TIMESTAMP=20
        self.ED_FUNC_ID=21
        self.ED_FUNC_VALUE=22
        self.ED_MARKER=23
        self.ED_SYNC_SIGNAL=24

        self.targetChannelList = [self.ED_RAW_CQ,self.ED_AF3, self.ED_F7, self.ED_F3, self.ED_FC5, self.ED_T7,self.ED_P7, self.ED_O1, self.ED_O2, self.ED_P8, self.ED_T8,self.ED_FC6, self.ED_F4, self.ED_F8, self.ED_AF4, self.ED_GYROX, self.ED_GYROY, self.ED_TIMESTAMP, self.ED_FUNC_ID, self.ED_FUNC_VALUE, self.ED_MARKER, self.ED_SYNC_SIGNAL]
        #self.header = ['COUNTER','AF3','F7','F3', 'FC5', 'T7', 'P7', 'O1', 'O2','P8', 'T8', 'FC6', 'F4','F8', 'AF4','GYROX', 'GYROY', 'TIMESTAMP','FUNC_ID', 'FUNC_VALUE', 'MARKER', 'SYNC_SIGNAL']
        #write = sys.stdout.write
        self.eEvent      = libEDK.EE_EmoEngineEventCreate()
        self.eState      = libEDK.EE_EmoStateCreate()
        self.userID            = c_uint(0)
        self.nSamples   = c_uint(0)
        self.nSam       = c_uint(0)
        self.nSamplesTaken  = pointer(self.nSamples)
        #self.da = zeros(128,double)
        self.data     = pointer(c_double(0))
        self.user     = pointer(self.userID)
        self.composerPort          = c_uint(1726)
        self.secs      = c_float(1)
        self.datarate    = c_uint(0)
        self.readytocollect    = False
        self.option      = c_int(0)
        self.state     = c_int(0)

        print libEDK.EE_EngineConnect("Emotiv Systems-5")
        if libEDK.EE_EngineConnect("Emotiv Systems-5") != 0:
            print "Emotiv Engine start up failed."

        print "Start receiving EEG Data! Press any key to stop logging...\n"
        #f = file(file_loc, 'w')
        #f = open(file_loc, 'w')
        #print >> f,header

        self.hData = libEDK.EE_DataCreate()
        libEDK.EE_DataSetBufferSizeInSec(self.secs)

        self.j=0

        print "Buffer size in secs:"

    def data_acq(self):
        while (1):
            #print('new samples')
            state = libEDK.EE_EngineGetNextEvent(self.eEvent)
            if state == 0:
                eventType = libEDK.EE_EmoEngineEventGetType(self.eEvent)
                libEDK.EE_EmoEngineEventGetUserId(self.eEvent, self.user)
                if eventType == 16: #libEDK.EE_Event_enum.EE_UserAdded:
                    #print "User added"
                    libEDK.EE_DataAcquisitionEnable(self.userID,True)
                    self.readytocollect = True
            
            if self.readytocollect==True:
                libEDK.EE_DataUpdateHandle(0, self.hData)
                libEDK.EE_DataGetNumberOfSample(self.hData,self.nSamplesTaken)
                #print "Updated :",self.nSamplesTaken[0]
                if self.nSamplesTaken[0] == 128:
                    self.nSam=self.nSamplesTaken[0]
                    arr=(ctypes.c_double*self.nSamplesTaken[0])()
                    ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
                    #libEDK.EE_DataGet(hData, 3,byref(arr), nSam)                         
                    data = array('d')#zeros(nSamplesTaken[0],double)
                    #print('yes')
                    y = np.zeros((128,14))
                    for sampleIdx in range(self.nSamplesTaken[0]):
                        x = np.zeros(14)
                        for i in range(1,15):
                            libEDK.EE_DataGet(self.hData,self.targetChannelList[i],byref(arr), self.nSam)
                            x[i-1] = arr[sampleIdx]
                            
                        y[sampleIdx] = x
                    #print(len(y)+345)
                    y = np.transpose(y)
                    
                    if (self.choice == 'A' or self.choice == 'B') and self.time<=12 and self.nSamplesTaken[0] == 128:
                        t = 0
                        t = self.time
                        
                        if self.time!=5 or self.time!=6 or self.time!=7:
                            if self.time>=8:
                                t = self.time-3
                            for sensor in range(0,14):
                                self.trial_data_row[0,((sensor*128)+(t*1792)):(((sensor*128)+127)+(t*1792))] = y[sensor,0:127]
                            self.time = self.time + 1

                            if t==9:
                                if self.choice=='A':
                                    self.trial_data_row[0,17920] = 1
                                elif self.choice == 'B':
                                    self.trial_data_row[0,17920] = 2
                                print('aaya')    
                                with file((self.profile_loc+self.profile_name+'.csv'),'a') as f:
                                    np.savetxt(f,self.trial_data_row,fmt='%f',delimiter=',',newline=' ')
                                    print>>f
                                    f.close()
                                    
                       
                    if self.time==0 or self.time==13:
                        self.choice = random.choice(['A','B'])
                        self.time = 0

                    if self.time>=0 and self.time<=4:
                        self.self_kivy.lbl3.text = 'Please Take a Rest'
                        self.self_kivy.text = os.getcwd()+'\\car_rest.jpg'
                    elif self.time>=5 and self.time<=7:
                        self.self_kivy.lbl3.text = 'Please Ready for Action'
                    elif self.time>=8 and self.time<=12:
                        if self.choice == 'A':
                            self.self_kivy.text = os.getcwd()+'\\car_left.png'
                            self.self_kivy.lbl3.text = 'Think about Left'
                        elif self.choice == 'B':
                            self.self_kivy.text = os.getcwd()+'\\car_Right.png'
                            self.self_kivy.lbl3.text = 'Think about Right'
                        
                else:
                    print('No')
            
            time.sleep(1.1)

            if self.exit==1:
                self.disconnect_engine()
                self.self_kivy.lbl3.text = 'Disconnected'
                print("Engine Disconnected")
                break

    def disconnect_engine(self):
        libEDK.EE_DataFree(self.hData)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------
        libEDK.EE_EngineDisconnect()
        libEDK.EE_EmoStateFree(self.eState)
        libEDK.EE_EmoEngineEventFree(self.eEvent)

    def disconnect(self):
        print('aaya disconnect karny k liye')
        self.exit=1

    
