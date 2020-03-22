
import keyboard
import PyQt5 as p
from PIL import Image as im
import numpy as np
import time
import copy
import sys
import matplotlib
import platform
import queue
import seaborn as sns
if platform.system() != 'Darwin' :
    matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QSize,pyqtSignal,QThread, QEvent, pyqtSlot

convergence_threshold = 2.0e-6
IMAGES_PATH = "images/"
CONFIGURATIONS_PATH = "configurations/"
q = queue.Queue(1)
q2 = queue.Queue(1)
q3 = queue.Queue(1)
c = [queue.Queue(1),queue.Queue(1)]
zeroState = None
zeroStatePrev = None
N_DET_LEVEL = 0.8
sys.setrecursionlimit(10000)
STOP=False
w_screen = 0
h_screen = 0
GUI=None
DETERMINISTIC=False
PROGRESSVALUE=0
AGENT=1
START_VALUE = -5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 68, 255)
GRAY = (135, 135, 125)
DONE = False # is our program newEnvironmentButtoned running?
ROUND = 1000
# create the WINDOW and CLOCK
labelGrid=[]
REMAKE=True
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 68, 255)
GRAY = (135, 135, 125)
MARGIN=1
WIDTH=0
TRAINING=False
START=[]
KIT=[]
WIN=[]
DETERMINISTICTEST=True
LOSE_STATE=[]
actions = ["up", "down", "left", "right"]
Q_Delta = []
Q_Delta.append([])
Q_Delta.append([])
Q_Delta[0].append([])
Q_Delta[0].append([])
Q_Delta[1].append([])
Q_Delta[1].append([])
GRID=[]
element={"ag":False,"kit":False,"win":False,"ag2":False,"kit2":False,"wall":True}
buttonclicked="null"
REALTIMEOBSTACLE=[[],False]
id=0
VELOCITY=None
TERMINATED=[False,False]
dialog=None
rewards = [ 0, -5 ]
PARAMETRI_COLORAZIONI = [[None,None,None,None],[None,None,None,None]]
TESTING = False
SU = -80
def handle_close(evt):
    mainWindow.setEnabled(True)

def resetImageSize(WIDTH):


    # im.open(IMAGES_PATH+"wino.jpg").resize((WIDTH,WIDTH),im.ANTIALIAS).save(IMAGES_PATH+"win.jpg")
    # im.open(IMAGES_PATH+"k1o.jpg").resize((WIDTH,WIDTH),im.ANTIALIAS).save(IMAGES_PATH+"k1.jpg")
    # im.open(IMAGES_PATH+"i1o.jpg").resize((WIDTH,WIDTH),im.ANTIALIAS).save(IMAGES_PATH+"i1.jpg")
    # im.open(IMAGES_PATH+"k2o.png").resize((WIDTH,WIDTH),im.ANTIALIAS).save(IMAGES_PATH+"k2.png")
    # im.open(IMAGES_PATH+"mo.jpg").resize((WIDTH,WIDTH),im.ANTIALIAS).save(IMAGES_PATH+"m.jpg")
    # im.open(IMAGES_PATH+"i2o.jpg").resize((WIDTH,WIDTH),im.ANTIALIAS).save(IMAGES_PATH+"i2.jpg")
    # im.open(IMAGES_PATH+"vico.png").resize((WIDTH,WIDTH),im.ANTIALIAS).save(IMAGES_PATH+"vic.png")
 
    global win,ag,kit,kit2,wall,whit,img,ag2,vic
    win=QPixmap(IMAGES_PATH+"wino.jpg").scaled(WIDTH,WIDTH,QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.SmoothTransformation)
    ag=QPixmap(IMAGES_PATH+"i1o.jpg").scaled(WIDTH,WIDTH,QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.SmoothTransformation)
    ag2=QPixmap(IMAGES_PATH+"i2o.jpg").scaled(WIDTH,WIDTH,QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.SmoothTransformation)
    kit=QPixmap(IMAGES_PATH+"k1o.jpg").scaled(WIDTH,WIDTH,QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.SmoothTransformation)
    kit2=QPixmap(IMAGES_PATH+"k2o.png").scaled(WIDTH,WIDTH,QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.SmoothTransformation)
    wall=QPixmap(IMAGES_PATH+"mo.jpg").scaled(WIDTH,WIDTH,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)
    whit=QPixmap(IMAGES_PATH+"white.jpg")#.scaled(WIDTH,WIDTH,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)
    vic=QPixmap(IMAGES_PATH+"vico.png").scaled(WIDTH,WIDTH,QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.SmoothTransformation)
    img={"win":win,"wall":wall,"kit":kit,"kit2":kit2,"ag":ag,"ag2":ag2,"vic":vic,"whit":whit}
 
class State:
    def __init__(self, state, win):
        self.board = np.zeros([ROWS, COLS])
        self.board[1, 1] = -1
        self.state = state
        self.win = win
        self.isEnd = False
        self.determine = DETERMINISTIC
 
    def giveReward(self):
        if self.state == self.win:
            return rewards[0]
        else:
            return rewards[1]
 
    def chooseActionProb(self, action,train,deleted):
        if len(deleted) == 0 :
            if action == "up":
                action = np.random.choice(["up", "left", "right"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
            elif action == "down":
                action = np.random.choice(["down", "left", "right"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
            elif action == "left":
                action = np.random.choice(["left", "up", "down"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
            elif action == "right":
                action = np.random.choice(["right", "up", "down"],   p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
        elif len(deleted) == 1 :
            if "up" in deleted : 
                if action == "down":
                    action = np.random.choice(["down","left", "right"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
                elif action == "left":
                    action = np.random.choice(["left", "down"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action == "right":
                    action = np.random.choice(["right", "down"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
            elif "down" in deleted : 
                if action == "up":
                    action = np.random.choice(["up","left", "right"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
                elif action == "left":
                    action = np.random.choice(["left", "up"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action == "right":
                    action = np.random.choice(["right", "up"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
            elif "right" in deleted :
                if action == "left":
                    action = np.random.choice(["left","up", "right"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
                elif action == "up":
                    action = np.random.choice(["up", "left"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action == "down":
                    action = np.random.choice(["doen", "left"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
            elif "left" in deleted : 
                if action == "right":
                    action = np.random.choice(["right","up", "down"],  p=[N_DET_LEVEL, (1-N_DET_LEVEL)/2, (1-N_DET_LEVEL)/2])
                elif action == "up":
                    action = np.random.choice(["up", "right"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action == "down":
                    action = np.random.choice(["down", "right"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])      
        elif len(deleted) == 2 :
            if "up" in deleted and "left" in deleted :
                if action == "right":
                    action = np.random.choice(["right", "down"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action =="down" :
                    action = np.random.choice(["down", "right"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
            elif "up" in deleted and "right" in deleted :
                if action == "left":
                    action = np.random.choice(["left", "down"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action =="down" :
                    action = np.random.choice(["down", "left"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
            elif "down" in deleted and "left" in deleted :
                if action == "right":
                    action = np.random.choice(["right", "up"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action =="down" :
                    action = np.random.choice(["up", "right"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
            elif "down" in deleted and "right" in deleted :
                if action == "left":
                    action = np.random.choice(["left", "up"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
                elif action =="down" :
                    action = np.random.choice(["up", "left"], p=[N_DET_LEVEL, (1-N_DET_LEVEL)])
        return action

    def isEndFunc(self):
        global DETERMINISTICTEST
        if (self.state == self.win) or (self.state in LOSE_STATE):
            self.isEnd = True
 
    def nxtPosition(self, action,train,deleted):
        if self.determine :
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
                if DETERMINISTIC == False :
                    self.determine= False
        else:
            self.determine= True
            if train or DETERMINISTICTEST:
                action = self.chooseActionProb(action,train,deleted)
            nxtState = self.nxtPosition(action,train,deleted)
        if (nxtState[0] >= 0) and (nxtState[0] <ROWS ):
            if (nxtState[1] >= 0) and (nxtState[1] < COLS):
                    if (nxtState[0],nxtState[1]) not in LOSE_STATE :
                            return nxtState
        return self.state
 
    def showBoard(self):
        self.board[self.state] = 1
        for i in range(ROWS):
            print('-----------------')
            out = '| '
            for j in range(COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == START_VALUE:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')

class Agent:
    def setEnvironment(self,j):
        if j == 0 :
            prev = WHITE
            GRID[START[self.id][0]][START[self.id][1]] = GREEN
            GRID[KIT[self.id][0]][KIT[self.id][1]]  = BLUE
        else:
            prev = BLUE
            GRID[START[self.id][0]][START[self.id][1]] = WHITE
            GRID[KIT[self.id][0]][KIT[self.id][1]]  = GREEN
        GRID[WIN[0][0]][WIN[0][1]]  = RED
        for l in LOSE_STATE:
            GRID[l[0]][l[1]]  = BLACK
        return prev
 
    def __init__(self,id,change_value,color):
        self.id=id
        self.deleted = []
        self.color = color
        self.change_value=change_value
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = {}
        self.State[0] = State(START[self.id],KIT[self.id])
        self.State[1] = State(KIT[self.id],WIN[0])
        self.lr = 0.5
        global DETERMINISTIC
        self.exp_rate = 0.3
        self.lr = 0.2
        self.exp_rate = 0.3
        self.gamma = 0.9
        self.goToWin = False
        self.decay_gamma = 0.9
        self.Q_prev = []
        self.Q_values = []
        self.Q_values.append({})
        self.Q_values.append({})
        self.Q_prev.append({})
        self.Q_prev.append({})
 
        self.state_values = {}
        self.state_values2 = {}
        for i in range(ROWS):
            for j in range(COLS):
                self.state_values[0,(i, j)] = START_VALUE  # set initial value to 0
                self.state_values[1,(i, j)] = START_VALUE  # set initial value to 0
                self.Q_values[0][(i, j)] = {}
                self.Q_values[1][(i, j)] = {}
                self.Q_prev[0][(i,j)] = {}
                self.Q_prev[1][(i,j)] = {}
                for a in actions :
                    #if (i==0 and a=="up") or (i==ROWS-1 and a== "down") or (j==0 and a=="left") or (j==COLS-1 and a=="right"):
                    #    continue
                    self.Q_values[0][(i, j)][a] = START_VALUE
                    self.Q_values[1][(i, j)][a] = START_VALUE
                    self.Q_prev[0][(i, j)][a] = START_VALUE
                    self.Q_prev[1][(i, j)][a] = START_VALUE
 
    def chooseAction(self,j,train,direct):
        global zeroState,zeroStatePrev
        mx_nxt_reward = float("-inf")
        action = ""
        actions_act = ["up", "left", "right","down"]
        if np.random.uniform(0, 1) <= self.exp_rate and train :
            while(not STOP):
                if direct=="up" :
                    action = np.random.choice(["up", "left", "right","down"], p=[0.4,0.2,0.2,0.2])
                if direct=="left" :
                    action = np.random.choice(["up", "left", "right","down"], p=[0.2,0.4,0.2,0.2])
                if direct=="right" :
                    action = np.random.choice(["up", "left", "right","down"], p=[0.2,0.2,0.4,0.2])
                if direct=="down" :
                    action = np.random.choice(["up", "left", "right","down"], p=[0.2,0.2,0.2,0.4])
                if (self.State[j].state[0]==0 and action=="up") or (self.State[j].state[0]==ROWS-1 and action== "down") or (self.State[j].state[1]==0 and action=="left") or (self.State[j].state[1]==COLS-1 and action=="right"):
                        continue
                else:
                    return action
        else:
            # greedy action
            if not train and self.id == 1 and zeroState != WIN and True not in TERMINATED:              
                if (self.State[j].state[0] - 1, self.State[j].state[1]) == zeroState :
                    if ("up") in actions_act :
                        actions_act.remove("up") 
                        self.deleted.append("up")
                if (self.State[j].state[0] + 1, self.State[j].state[1]) == zeroState :
                    if ("down") in actions_act :
                        actions_act.remove("down")
                        self.deleted.append("down")                       
                if (self.State[j].state[0], self.State[j].state[1]-1) == zeroState :
                    if ("left") in actions_act :
                        actions_act.remove("left")
                        self.deleted.append("left")
                if (self.State[j].state[0], self.State[1].state[1]+1) == zeroState:
                    if ("right") in actions_act :
                        actions_act.remove("right")
                        self.deleted.append("right")                
                if (self.State[j].state[0] - 1, self.State[j].state[1]) == zeroStatePrev and zeroState ==  (self.State[j].state[0], self.State[j].state[1]):
                    if ("up") in actions_act :
                        actions_act.remove("up")
                        self.deleted.append("up")
                if (self.State[j].state[0] + 1, self.State[j].state[1]) == zeroStatePrev and zeroState ==  (self.State[j].state[0], self.State[j].state[1]):              
                    if ("down") in actions_act :
                        actions_act.remove("down")
                        self.deleted.append("down")
                if (self.State[j].state[0], self.State[j].state[1]-1) == zeroStatePrev and zeroState ==  (self.State[j].state[0], self.State[j].state[1]):
                    if ("left") in actions_act :
                        actions_act.remove("left")
                        self.deleted.append("left")
                if (self.State[j].state[0], self.State[j].state[1]+1) == zeroStatePrev and zeroState ==  (self.State[j].state[0], self.State[j].state[1]):
                    if ("right") in actions_act :
                        actions_act.remove("right")  
                        self.deleted.append("right")          
            if (self.State[j].state[0] - 1, self.State[j].state[1]) in LOSE_STATE or (self.State[j].state[0]==0 ) :
                if ("up") in actions_act :
                    actions_act.remove("up")
                    self.deleted.append("up")
            if (self.State[j].state[0] + 1, self.State[j].state[1]) in LOSE_STATE or (self.State[j].state[0]==ROWS-1 ) :
                if ("down") in actions_act :
                    actions_act.remove("down")
                    self.deleted.append("down")
            if (self.State[j].state[0], self.State[j].state[1]-1) in LOSE_STATE or (self.State[j].state[1]==0 ):
                if ("left") in actions_act :
                    actions_act.remove("left")
                    self.deleted.append("left")
            if (self.State[j].state[0], self.State[j].state[1]+1) in LOSE_STATE or (self.State[j].state[1]==COLS-1 ):
                if ("right") in actions_act :
                    actions_act.remove("right")
                    self.deleted.append("right")
            for a in actions_act :
                current_position = self.State[j].state
                nxt_reward = self.Q_values[j][current_position][a]
                if nxt_reward > mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action
     
    def takeAction(self, action,j,train,deleted):
        position = self.State[j].nxtPosition(action,train,deleted)
        if j == 0:
            return State(state=position, win=KIT[self.id])
        elif j == 1:
            return State(state=position,win= WIN[0])
 
    def reset(self,j):
        self.states = []
        if j == 0 :
            self.State[0] = State(state=START[self.id],win =KIT[self.id])
        elif j == 1 :
            self.State[1] = State(state=KIT[self.id],win=WIN[0])
 
    def playTD(self, rounds,j,train):
        global GRID,PROGRESSVALUE,zeroState,zeroStatePrev,TERMINATED,PARAMETRI_COLORAZIONI,LOSE_STATE,convergence_threshold
        passi = 0
        passiDelta = 0
        i = 0
        conv = 0
        direct = "right"
        self.exp_rate = 0.3
        prev=self.setEnvironment(j)
        r1=c1=r2=c2=None
        sommaas = 0
        enne=0
        first_time = True
        while i < rounds and not STOP:
            passiDelta = 0
            self.Q_prev = copy.deepcopy(self.Q_values)
            while True and not STOP:
                passiDelta+=1
                if self.State[j].isEnd :
                    Q_Delta[self.id][j].append(passiDelta)
                    if self.State[j].giveReward() == 0 :
                        self.exp_rate = self.exp_rate * 0.99997
                        if train:
                            self.change_value.emit(PROGRESSVALUE)
                            PROGRESSVALUE+=1
                        i += 1
                    end = 0 
                    for s1 in range(ROWS):
                        for s2 in range(COLS) :
                            for a in self.actions : 
                                sommaas+=(self.Q_values[j][(s1,s2)][a] - self.Q_prev[j][(s1,s2)][a])**2
                                enne+=1    
                    sommaas = sommaas/enne
                    if  sommaas < convergence_threshold and convergence_threshold != -1:
                        conv+=1
                        if conv == 10 :
                            i = rounds
                    else : 
                        conv = 0
                   
                    self.Q_prev = copy.deepcopy(self.Q_values)
                    prev = self.setEnvironment(j)
                    self.reset(j)
                    break
                if not train and self.id == 1 and AGENT == 2 and True not in TERMINATED:
                    q.get()
                self.deleted.clear()
                action = self.chooseAction(j,train,direct)
                finalState = self.takeAction(action,j,train,self.deleted)
                if not train and self.id == 0 and AGENT == 2 and True not in TERMINATED:
                    zeroStatePrev = (self.State[j].state[0],self.State[j].state[1])
                    zeroState = (finalState.state[0],finalState.state[1])
                    q.put(1)
                if not train and self.id == 1 and AGENT == 2  and True not in TERMINATED:
                    q2.put(1)
                if not train and self.id == 0 and AGENT == 2  and True not in TERMINATED:
                    q2.get()
                if finalState != self.State :
                    passi = passi + 1
                    if passi == 100 :
                        if direct == "right" :
                            direct = "left"
                        elif direct == "left" :
                            direct = "up"
                        elif direct == "up" :
                            direct = "down"
                        else :
                            direct = "right"
                        passi = 0
                    reward = finalState.giveReward()
                    finalState.isEndFunc()
                    GRID[self.State[j].state[0]][self.State[j].state[1]] = prev
                    r1=self.State[j].state[0]
                    c1=self.State[j].state[1]
                    if train :
                        maximum = float("-inf")
                        for a in self.actions :
                            if (finalState.state[0]==0 and a=="up") or (finalState.state[0]==ROWS-1 and a== "down") or (finalState.state[1]==0 and a=="left") or (finalState.state[1]==COLS-1 and a=="right"):
                                    continue
                            cur = self.Q_values[j][finalState.state][a]
                            if cur > maximum :
                                maximum = cur
                        actual = round(self.Q_values[j][self.State[j].state][action] + self.lr*(reward + self.gamma*maximum - self.Q_values[j][self.State[j].state][action]),3)
                        self.Q_values[j][self.State[j].state][action] = actual
                    prev = GRID[finalState.state[0]][finalState.state[1]]
                    GRID[finalState.state[0]][finalState.state[1]] = GREEN
                    if not train and self.id == 0 and AGENT == 2 and True not in TERMINATED:
                        q3.get()
                    if not train :
                        PARAMETRI_COLORAZIONI[self.id] = [self.State[j].state[0],self.State[j].state[1],finalState.state[0],finalState.state[1]]
                        if AGENT == 1 :
                            self.color.emit(PARAMETRI_COLORAZIONI)
                        else:
                            c[self.id].put(PARAMETRI_COLORAZIONI[self.id])
                        if AGENT == 1 :
                            if VELOCITY.value() > 10 : 
                                time.sleep(1/(100))
                            elif VELOCITY.value() == 1 : 
                                time.sleep(2)
                            else :     
                                time.sleep(1/(VELOCITY.value()+2))
                        
                    if not train and self.id == 1 and AGENT == 2 and True not in TERMINATED:
                        q3.put(1)
                    self.State[j] = finalState
                    r2=self.State[j].state[0]
                    c2=self.State[j].state[1]
            if not STOP and train:
                self.change_value.emit(PROGRESSVALUE)
                
            if not train:
                break      
    def saveValues(self):
        np.save(CONFIGURATIONS_PATH+'stateValuesQ'+str(self.id)+'.npy',self.Q_values)
        np.save(CONFIGURATIONS_PATH+"environment.npy",np.asarray([AGENT,ROWS,COLS,START,WIN,KIT,LOSE_STATE,rewards,ROUND,convergence_threshold]))
       
    def readValues(self) :
        self.Q_values = np.load(CONFIGURATIONS_PATH+'stateValuesQ'+str(self.id)+'.npy',allow_pickle=True)
 
    def showValues(self,k):
        for i in range(ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(COLS):
                if self.State[0].determine:
                    out += str(self.state_values[k,(i, j)]).ljust(6) + ' | '
                else:
                    for a in actions:
                        if (i==0 and a=="up") or (i==ROWS-1 and a== "down") or (j==0 and a=="left") or (j==COLS-1 and a=="right"):
                            continue
                        out += str(self.Q_values[k][i, j][a]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')
              
class QLabel_Clicked(QtWidgets.QLabel):
    on = pyqtSignal(list)
    def __init__(self, parent,i,j):
        QtWidgets.QLabel.__init__(self,parent)
        self.i=i
        self.j=j
           
    def eventFilter(self, object, event):
        global buttonclicked,WIN,KIT,START,GRID,LOSE_STATE,element
        global TERMINATED
        post=None
        if event.type()==QEvent.MouseButtonPress :
            a=labelGrid[self.i][self.j][1]
            if buttonclicked != "wall":
                 
                if buttonclicked!="null":
                    if a == "wall":
                        LOSE_STATE.remove((self.i,self.j))
                    if buttonclicked=="win":
                        post="ag"
                        GRID[self.i][self.j] = RED
                        WIN = [(self.i,self.j)]
                        self.parentWidget().win.setEnabled(False)
                        self.parentWidget().win_label.setText("WIN added")
                    elif buttonclicked=="kit":
                        post="ag2"
                        GRID[self.i][self.j] = BLUE
                        KIT[0]=(self.i,self.j)
                        self.parentWidget().kit.setEnabled(False)
                        self.parentWidget().kit_label.setText("KIT added")
                    elif buttonclicked=="ag":
                        post="kit"
                        GRID[self.i][self.j] = GREEN
                        START[0] = (self.i,self.j)
                        self.parentWidget().ag.setEnabled(False)
                        self.parentWidget().ag_label.setText("AGENT added")
                    elif buttonclicked=="ag2":
                        post="kit2"
                        GRID[self.i][self.j] = GREEN
                        START[1] = (self.i,self.j)
                        self.parentWidget().ag2.setEnabled(False)
                        self.parentWidget().ag2_label.setText("AGENT 2 added")
                    elif buttonclicked=="kit2":
                        post="wall"
                        GRID[self.i][self.j] = BLUE
                        KIT[1]=(self.i,self.j)
                        self.parentWidget().kit2.setEnabled(False)
                        self.parentWidget().kit2_label.setText("KIT 2 added")    
                    
                    if a== "win":
                        GRID[self.i][self.j] = WHITE
                        self.parentWidget().win.setEnabled(True)
                        self.parentWidget().win_label.setText("Add WIN")
                    elif a== "kit":
                        GRID[self.i][self.j] = WHITE
                        KIT[0]=0
                        self.parentWidget().kit.setEnabled(True)
                        self.parentWidget().kit_label.setText("Add KIT")
                    elif a== "ag":
                        GRID[self.i][self.j] = WHITE
                        START[0]=0
                        self.parentWidget().ag.setEnabled(True)
                        self.parentWidget().ag_label.setText("Add AGENT")
                    elif a== "ag2":
                        GRID[self.i][self.j] = WHITE
                        START[1]=0
                        self.parentWidget().ag2.setEnabled(True)
                        self.parentWidget().ag2_label.setText("Add AGENT 2")
                    elif a== "kit2":
                        GRID[self.i][self.j] = WHITE
                        KIT[1]=0
                        self.parentWidget().kit2.setEnabled(True)
                        self.parentWidget().kit2_label.setText("Add KIT 2")
                   
                    labelGrid[self.i][self.j][1]= buttonclicked
                    element[labelGrid[self.i][self.j][1]]= True
                    labelGrid[self.i][self.j][0].setPixmap(img[buttonclicked])
                   
                    if buttonclicked!="wall":
                        buttonclicked="null"
                        #buttonclicked = post
                else:
                    a=labelGrid[self.i][self.j][1]
                    labelGrid[self.i][self.j][0].setPixmap(whit)
                    if a=="win":
                        self.parentWidget().win_label.setText("Add WIN")
                        self.parentWidget().win.setEnabled(True)
                    elif a== "kit":
                        self.parentWidget().kit.setEnabled(True)
                        self.parentWidget().kit_label.setText("Add KIT")
                    elif a== "ag":
                        self.parentWidget().ag.setEnabled(True)
                        self.parentWidget().ag_label.setText("Add AGENT")
                    elif a== "kit2":
                        self.parentWidget().kit2.setEnabled(True)
                        self.parentWidget().kit2_label.setText("Add KIT 2")
                    elif a== "ag2":
                        self.parentWidget().ag2.setEnabled(True)
                        self.parentWidget().ag2_label.setText("Add AGENT 2")
                
                    element[a]=False
                    labelGrid[self.i][self.j][1]="whit"
                if element["ag"] and element["kit"] and element["win"] and not element["kit2"] and not element["ag2"]:
                    self.parentWidget().ag2.setEnabled(True)
                    self.parentWidget().kit2.setEnabled(True)
                    self.parentWidget().trainButton.setEnabled(True)
                    self.parentWidget().reward_button.setEnabled(True)
                    self.parentWidget().choose_train_button.setEnabled(True)
                elif element["ag"] and element["kit"] and element["win"] and element["kit2"] and element["ag2"]:
                    self.parentWidget().trainButton.setEnabled(True)
                    self.parentWidget().reward_button.setEnabled(True)
                    self.parentWidget().choose_train_button.setEnabled(True)
                else:
                    self.parentWidget().trainButton.setEnabled(False)
                    self.parentWidget().reward_button.setEnabled(False)
                    self.parentWidget().choose_train_button.setEnabled(False)
            elif (a == "whit"):
                LOSE_STATE.append((self.i,self.j))
                labelGrid[self.i][self.j][1]= buttonclicked
                labelGrid[self.i][self.j][0].setPixmap(img[buttonclicked])
 
        elif event.type() == 10 and (labelGrid[self.i][self.j][1] == "whit" or labelGrid[self.i][self.j][1] == "wall"):#Mouse on label
 
            if (keyboard.is_pressed('m') or keyboard.is_pressed('ctrl') and  labelGrid[self.i][self.j][1] == "whit"):
                LOSE_STATE.append((self.i,self.j))
                self.on.emit([self.i,self.j,"wall"])
            elif (keyboard.is_pressed('alt') or keyboard.is_pressed('c')) and labelGrid[self.i][self.j][1]=="wall":
                LOSE_STATE.remove((self.i,self.j))
                self.on.emit([self.i,self.j,"whit"])
            return True
        return False
 
class MyT(QThread):
    change_value = pyqtSignal(int)
    global q
    end = pyqtSignal(int)
    color = pyqtSignal(list)
    def __init__(self,id):
        self.id= id
        QThread.__init__(self)
    def run(self):
        global ROUND
        global DETERMINISTIC
        global TERMINATED
        print ("AGENTE "+ str(self.id+1)+" START")
        ag = Agent(self.id,self.change_value,self.color)
        if not TRAINING:
            ag.readValues()
        ag.playTD(rounds=ROUND,j=0,train=TRAINING)
        ag.playTD(rounds=ROUND,j=1,train=TRAINING)
        if TRAINING:
            ag.saveValues()
        self.end.emit(self.id)          
        print ("AGENTE "+ str(self.id+1)+" END")
   
class ShowTestMultiAgent(QThread):
    end = pyqtSignal()
    def run(self):
        global REALTIMEOBSTACLE
        REALTIMEOBSTACLE[1]=True
        self.end.emit()
 
class Train(QThread):
    dialog = pyqtSignal()
    end = pyqtSignal()
    def run(self):
        self.dialog.emit()
        self.end.emit()
 
class ShowTestSingleAgent(QThread):
    end = pyqtSignal()
    color = pyqtSignal(list)
    def run(self):
        global REALTIMEOBSTACLE
        self.finito=[False,False]
        self.catch=[False,False]
        REALTIMEOBSTACLE[1]=True
        self.end.emit()
        tmp={0:"ag",1:"ag2"}
        tmp1={0:"kit",1:"kit2"}
        while not STOP:
            param=[[0,0,0,0],[0,0,0,0]]
            if VELOCITY.value() == 15 : 
                time.sleep(1/(100))
            elif VELOCITY.value() == 1 : 
                time.sleep(2)
            else :     
                time.sleep(1/(VELOCITY.value()+2))   
            if self.finito[0] and not self.finito[1]:
                param[1] = c[1].get()
                param[0][0] = WIN[0][0]
                param[0][1] = WIN[0][1]
                param[0][2] = WIN[0][0]
                param[0][3] = WIN[0][1]
            elif self.finito[1] and not self.finito[0]:
                param[0] = c[0].get()
                param[1][0] = WIN[0][0]
                param[1][1] = WIN[0][1]
                param[1][2] = WIN[0][0]
                param[1][3] = WIN[0][1]
            elif not self.finito[0] and not self.finito[1]:
                param[0] = c[0].get()
                param[1] = c[1].get()  
            elif  self.finito[0] and  self.finito[1]:
                break
            if param[0][2] == KIT[0][0] and param[0][3] == KIT[0][1] :
                    self.catch[0] = True
            if param[0][2] == WIN[0][0] and param[0][3] == WIN[0][1] and self.catch[0]:
                self.finito[0] = True
            if param[1][2] == KIT[1][0] and param[1][3] == KIT[1][1] :
                    self.catch[1] = True
            if param[1][2] == WIN[0][0] and param[1][3] == WIN[0][1] and self.catch[1]:
                self.finito[1] = True
 
            self.color.emit(param)          

class Ui(QtWidgets.QMainWindow):
    def close_UI(self,istance):
        istance[0].close()
        self.info_label.setText("Params Used:\n        Reward Final State = "+str(rewards[0])+"        Reward Not Fianl State = "+str(rewards[1])+"\n        Episodies = "+str(ROUND)+ "        Convergence Threshold = "+str(convergence_threshold))
 
    def choose_reward(self):
        tmp1 = Rewards_UI(self)
        tmp1.end.connect(self.close_UI)
        tmp1.show()

    def choose_training(self) : 
        tmp = Training_Settings_UI(self)
        tmp.end.connect(self.close_UI)
        tmp.show()

    def goThreads(self):
        q.queue.clear()
        q2.queue.clear()
        q3.queue.clear()
        c[0].queue.clear()
        c[1].queue.clear()
        global AGENT, ROUND,TERMINATED,N_DET_LEVEL
        N_DET_LEVEL=self.ndet_level.value()/100
        self.finito=[False,False]
        if START[1]!=0:
            AGENT=2
        else:
            AGENT = 1
        TERMINATED=[False,False]
        if TRAINING:    
            self.progressBar.setMaximum(ROUND*2*AGENT+100)
        if AGENT == 2:
            self.finito=[False,False]
            self.catch=[False,False]
            self.t2.start()
            self.t1.start()
        else:
            self.finito=False
            self.catch=False
            TERMINATED=[False,True]
            self.t1.start()
   
    def screenShot(self):
        mult = 1
        if platform.system() != 'Darwin' :
            s=QtWidgets.QApplication.primaryScreen().grabWindow(self.winId())
            measure = [labelGrid[0][0][0].geometry().x(), labelGrid[0][0][0].geometry().y(),int(WIDTH*self.rows.value()*w_screen),int(WIDTH*self.cols.value()*h_screen)]
            rect= QtCore.QRect(measure[0], measure[1], measure[2], measure[3])
            cropped = s.copy(rect)
            cropped = cropped.scaled(int(250*w_screen),int(250*w_screen))
            cropped.save(IMAGES_PATH+"screenshot","png")
            self.last.setPixmap(cropped)    
        else :
            mult=2
            s=QtWidgets.QApplication.primaryScreen().grabWindow(self.winId())
            s.save(IMAGES_PATH+"banana.jpg","jpeg")
            im.open(IMAGES_PATH+"banana.jpg").crop((int(480*w_screen*mult),int(120*h_screen*mult),int(1060*w_screen*mult),int(690*h_screen*mult))).resize((300,300),im.ANTIALIAS).save(IMAGES_PATH+"screenshot.jpg")  
            self.last.setPixmap(QPixmap(IMAGES_PATH+"screenshot.jpg"))
         
    def dialog(self):
        self.setEnabled(False)
        self.progressBar=QtWidgets.QProgressDialog(parent=self,labelText="Training...")
        if platform.system() == 'Darwin' :
            self.progressBar.setWindowModality(QtCore.Qt.WindowModal)
        self.progressBar.setVisible(True)
        self.progressBar.setAutoClose(True)
        self.progressBar.setValue(0)
        self.progressBar.setWindowTitle("Training")
        global progress
        progress = self.progressBar
        b1= QtWidgets.QPushButton("Stop",self.progressBar)
        b1.clicked.connect(self.stopLearning)
        b1.move(20,60)
        self.progressBar.setCancelButton(b1)
   
    def colorTest(self,param):
        pre_ag = param[0][0:2]
        curr_ag = param[0][2:4]
        pre_ag2 = param[1][0:2]
        curr_ag2 = param[1][2:4]
        kit0 = [KIT[0][0],KIT[0][1]]
        kit1 = [KIT[1][0],KIT[1][1]]
        win0 = [WIN[0][0],WIN[0][1]]
        if curr_ag == kit0:
            self.catch[0] = True
        elif curr_ag == win0 and self.catch[0]:
            self.finito[0] = True
        if curr_ag2 == kit1 :
            self.catch[1] = True
        elif curr_ag2 == win0 and self.catch[1]:
            self.finito[1]=True
 
        if not self.catch[0] and not self.catch[1]:
            if pre_ag == win0:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["win"])
            elif pre_ag == kit1:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["kit2"])
            elif pre_ag != curr_ag2:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(whit)
            if pre_ag2 == win0:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(img["win"])
            elif pre_ag2 == kit0:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(img["kit"])
            elif pre_ag2 != curr_ag:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(whit)
        elif self.catch[0] and not self.catch[1]:
            if pre_ag == win0:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["win"])
            elif pre_ag == kit1:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["kit2"])
            elif pre_ag != curr_ag2:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(whit)
   
            if pre_ag2 == win0:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(img["win"])
            elif pre_ag2 != curr_ag:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(whit)
 
        elif not self.catch[0] and self.catch[1]:
 
            if pre_ag == WIN[0]:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["win"])
            # elif pre_ag[0] == kit0[0] and pre_ag[1] == kit0[1]:
            #     labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["kit"])
            elif pre_ag != curr_ag2:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(whit)
           
            if pre_ag2 == win0:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(img["win"])
            elif pre_ag2 == kit0:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(img["kit"])
            elif pre_ag2 != curr_ag:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(whit)
 
        elif self.catch[0] and self.catch[1]:
            if pre_ag == win0:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["win"])
            elif pre_ag != curr_ag2:
                labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(whit)
            if pre_ag2 == win0 :
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(img["win"])
            elif pre_ag2 != curr_ag:
                labelGrid[pre_ag2[0]][pre_ag2[1]][0].setPixmap(whit)
           
        if False not in self.finito:
            labelGrid[WIN[0][0]][WIN[0][1]][0].setPixmap(vic)
        else:
           
            labelGrid[curr_ag[0]][curr_ag[1]][0].setPixmap(img["ag"])
            labelGrid[curr_ag2[0]][curr_ag2[1]][0].setPixmap(img["ag2"])
        labelGrid[pre_ag[0]][pre_ag[1]][0].repaint()
        labelGrid[curr_ag[0]][curr_ag[1]][0].repaint()
        labelGrid[pre_ag2[0]][pre_ag2[1]][0].repaint()
        labelGrid[curr_ag2[0]][curr_ag2[1]][0].repaint()
    
    def repaintCurrPre(self,param):
        nparam=param[1]
        param=param[0]
        labelGrid[param[0]][param[1]][0].repaint()
        labelGrid[param[2]][param[3]][0].repaint()
        if nparam == 8:
            labelGrid[param[4]][param[5]][0].repaint()
            labelGrid[param[6]][param[7]][0].repaint()
   
    def endTest(self):
        self.allWidgets(True)
        self.trainButton.setEnabled(True)
        self.reward_button.setEnabled(True)
        self.choose_train_button.setEnabled(True)
        self.finito=[False,False]
        self.catch=[False,False]
 
    def allWidgets(self,mode):
        self.rows.setEnabled(mode)
        self.cols.setEnabled(mode)
        self.ndet_level.setEnabled(mode)
        self.reward_button.setEnabled(mode)
        self.choose_train_button.setEnabled(mode)
        self.newEnvironmentButton.setEnabled(mode)
        self.test.setEnabled(mode)
        self.velocity.setEnabled(mode)
        self.plot.setEnabled(mode)
        self.trainButton.setEnabled(mode)
        self.setButtonEnable(mode)
        self.last.setEnabled(mode)
   
    def setButtonEnable(self,mode):
        self.ag.setEnabled(mode)
        self.win.setEnabled(mode)
        self.kit.setEnabled(mode)
        self.ag2.setEnabled(mode)
        self.kit2.setEnabled(mode)
        self.wall.setEnabled(mode)
   
    def endThread(self,id):
        global TERMINATED, TESTING, PROGRESSVALUE, STOP
        TERMINATED[id]=True
        if TRAINING:
            if False not in TERMINATED:
                self.progressBar.close()
                self.setEnabled(True)
                global STOP
                STOP = False
                PROGRESSVALUE = 0
                plt.subplot(2,2,1)
                plt.title('A: KIT')
                plt.plot(Q_Delta[0][0])
                plt.subplot(2,2,2)
                plt.title('A: WIN')
                plt.plot(Q_Delta[0][1])
                if AGENT==2:
                    plt.subplot(2,2,3)
                    plt.title('B: KIT')
                    plt.plot(Q_Delta[1][0])
                    plt.subplot(2,2,4)
                    plt.title('B: WIN')
                    plt.plot(Q_Delta[1][1])
                    plt.tight_layout()
                plt.savefig(IMAGES_PATH+"grafico1.png")
                plt.close()
                plt.subplot(2,2,1)
                my_new_list = [i * rewards[1] for i in Q_Delta[0][0]]
                my_new_list2 = [i * rewards[1] for i in Q_Delta[0][1]]
                plt.title('A REWARDS')
                plt.plot([x+y for x,y in zip(my_new_list,my_new_list2)])
                if AGENT==2:
                    plt.subplot(2,2,2)
                    my_new_list = [i * rewards[1] for i in Q_Delta[1][0]]
                    my_new_list2 = [i * rewards[1] for i in Q_Delta[1][1]]
                    plt.title('B REWARDS')
                    plt.plot([x+y for x,y in zip(my_new_list,my_new_list2)])
                    plt.tight_layout()
                plt.savefig(IMAGES_PATH+"grafico2.png")
                plt.close()
                self.setButtonEnable(False)
                
        elif False not in TERMINATED:
            STOP = False
            self.allWidgets(True)
            self.trainButton.setEnabled(False)
            self.reward_button.setEnabled(False)
            self.choose_train_button.setEnabled(False)
            self.test.setText("Test last config")
            self.test.setEnabled(True)
            self.plot.setEnabled(True)
            TESTING = False
        else :
            if q.empty() :
                q.put(1)
            else :
                q.get()
            if q2.empty :
                q2.put(1)
            else :
                q2.get()
 
    def colorSingle(self,param):
        pre_ag = param[0][0:2]
        curr_ag = param[0][2:4]
        kit0 = [KIT[0][0],KIT[0][1]]
        win0 = [WIN[0][0],WIN[0][1]]
        if curr_ag == kit0:
            self.catch = True
        if curr_ag == win0 and self.catch:
            self.finito = True
        if pre_ag == win0 and not self.catch:
            labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(img["win"])
        else:
            labelGrid[pre_ag[0]][pre_ag[1]][0].setPixmap(whit)
        if self.finito == True:
            labelGrid[curr_ag[0]][curr_ag[1]][0].setPixmap(vic)
        else:
            labelGrid[curr_ag[0]][curr_ag[1]][0].setPixmap(img["ag"])
                     
    def __init__(self):
        super(Ui, self).__init__()
        global ROUND, mainWindow, w_screen, h_screen, WIDTH, screen, id
        mainWindow=self
        self.pressed=False
        self.finito=[False,False]
        self.catch=[False,False]
        self.setWindowTitle("How We Q_Learn")
        screen = QtWidgets.QDesktopWidget().screenGeometry(-1)
        w_screen = screen.width()/1366
        h_screen = screen.height()/768
        x=int((15+445+5+560+5)*w_screen)
        y=int((108+5)*h_screen)
        WIDTH=int(50*w_screen)
        resetImageSize(WIDTH)
    #FRAME
        self.frame=QtWidgets.QGroupBox(self)
        self.frame.setGeometry(int(15*w_screen),int(SU+y),int(440*w_screen),int(150*h_screen))
        self.frame.setStyleSheet("border: 5px solid darkGray")
        
        self.frame=QtWidgets.QGroupBox(self)
        self.frame.setGeometry(int(15*w_screen),SU+int((95+180)*h_screen),int(440*w_screen),int(320*h_screen))
        self.frame.setStyleSheet("border: 5px solid darkGray")
        self.frame=QtWidgets.QGroupBox(self)
        self.frame.setGeometry(int(15*w_screen),SU+int((95+170+1)*h_screen)+int(340*h_screen),int(440*w_screen),int(115*h_screen))
        self.frame.setStyleSheet("border: 5px solid darkGray")
        self.frame=QtWidgets.QGroupBox(self)
        self.frame.setGeometry(int((15+445)*w_screen),SU+y,int(560*w_screen),int(560*h_screen))
        self.frame.setStyleSheet("border: 5px solid darkGray")
        self.frame=QtWidgets.QGroupBox(self)
        self.frame.setGeometry(int(x-5*h_screen),SU+y,int(330*w_screen),int(800*h_screen/4.5))
        self.frame.setStyleSheet("border: 5px solid darkGray")
        self.frame=QtWidgets.QGroupBox(self)
        self.frame.setGeometry(int(x-5*h_screen),SU+int(y+5*h_screen+int(800*h_screen/4.5)),int(330*w_screen),int(800*h_screen/4.5))
        self.frame.setStyleSheet("border: 5px solid darkGray")
        self.frame=QtWidgets.QGroupBox(self)
        self.frame.setGeometry(int(x-5*h_screen),SU+int(y+(5*h_screen+int(800*h_screen/4.5))*2),int(330*w_screen),int(800*h_screen/4.5))
        self.frame.setStyleSheet("border: 5px solid darkGray")
    #BUTTON
        self.reward_button = QtWidgets.QPushButton(self)
        self.reward_button.setGeometry(int((230+90)*w_screen),SU+int(140*h_screen),int(90*w_screen),int(30*h_screen))
        self.reward_button.setText("Choose Reward")
        self.reward_button.clicked.connect(self.choose_reward)
        self.reward_button.setEnabled(False)
        self.choose_train_button = QtWidgets.QPushButton(self)
        self.choose_train_button.setGeometry(int((220)*w_screen),SU+int(140*h_screen),int(90*w_screen),int(30*h_screen))
        self.choose_train_button.setText("Train Settings")
        self.choose_train_button.clicked.connect(self.choose_training)
        self.choose_train_button.setEnabled(False)
        
        self.ag=QtWidgets.QPushButton(self)
        self.ag.setGeometry(x+80,SU+int(y+(WIDTH+40*h_screen)*3+20*h_screen),WIDTH,WIDTH)
        self.ag.clicked.connect(self.agclick)
        self.ag.setIcon(QIcon(QPixmap(ag)))
        self.ag.setIconSize(QSize(WIDTH,WIDTH))
        self.win=QtWidgets.QPushButton(self)
        self.win.setGeometry(x+80,SU+int(y+30),WIDTH,WIDTH)
        self.win.clicked.connect(self.winclick)
        self.win.setIcon(QIcon(win))
        self.win.setIconSize(QSize(WIDTH,WIDTH))

        self.kit=QtWidgets.QPushButton(self)
        self.kit.setGeometry(x+80,SU+int(y+(WIDTH+40*h_screen)*2+20*h_screen)+10,WIDTH,WIDTH)
        self.kit.clicked.connect(self.kitclick)
        self.kit.setIcon(QIcon(kit))
        self.kit.setIconSize(QSize(WIDTH,WIDTH))

        self.ag2=QtWidgets.QPushButton(self)
        self.ag2.setGeometry(x+80,SU+int(y+(WIDTH+40*h_screen)*5+20*h_screen),WIDTH,WIDTH)
        self.ag2.clicked.connect(self.ag2click)
        self.ag2.setEnabled(False)
        self.ag2.setIcon(QIcon(ag2))
        self.ag2.setIconSize(QSize(int(50*w_screen),int(50*h_screen)))
        
        self.wall=QtWidgets.QPushButton(self)
        self.wall.setGeometry(x+80,SU+int(y+(WIDTH+40*h_screen)+20*h_screen),WIDTH,WIDTH)
        self.wall.clicked.connect(self.wallclick)
        self.wall.setIcon(QIcon(wall))
        self.wall.setIconSize(QSize(WIDTH,WIDTH))
       
        self.kit2=QtWidgets.QPushButton(self)
        self.kit2.setGeometry(x+80,SU+int(y+(WIDTH+40*h_screen)*4+20*h_screen)+10,WIDTH,WIDTH)
        self.kit2.clicked.connect(self.kit2click)
        self.kit2.setIcon(QIcon(kit2))
        self.kit2.setIconSize(QSize(int(50*w_screen),int(50*h_screen)))

        self.trainButton=QtWidgets.QPushButton(self)
        self.trainButton.clicked.connect(self.train)
        self.trainButton.setGeometry(int(240*w_screen),SU+int(220*h_screen),int(200*w_screen),int(30*h_screen))
        self.trainButton.setText("Train")
        self.trainButton.setEnabled(False)
        
        self.newEnvironmentButton = QtWidgets.QPushButton(self)
        self.newEnvironmentButton.setText("Configure new environment")
        self.newEnvironmentButton.clicked.connect(self.newEnvironment)
        self.newEnvironmentButton.setGeometry(int(30*w_screen),SU+int(220*h_screen),int(200*w_screen),int(30*h_screen))

        self.load = QtWidgets.QPushButton(self)
        self.load.setText("Load last config")
        self.load.clicked.connect(self.initalize)
        self.load.setGeometry(int(40*w_screen),SU+int(620*h_screen),int(130*w_screen),int(30*h_screen))

        self.test = QtWidgets.QPushButton(self)
        self.test.setText("Test last config")
        self.test.clicked.connect(self.testLast)
        self.test.setGeometry(int(40*w_screen)+int(130*w_screen),SU+int(620*h_screen),int(130*w_screen),int(30*h_screen))

        self.plot = QtWidgets.QPushButton(self)
        self.plot.setText("Plot")
        self.plot.clicked.connect(self.showPlot)
        self.plot.setGeometry(int(40*w_screen)+int(130*w_screen)*2,SU+int(620*h_screen),int(130*w_screen),int(30*h_screen))

    #LABEL_BUTTON
        self.info_label = QtWidgets.QLabel(self)
        self.info_label.setFont(QtGui.QFont("Times", 13, QtGui.QFont.Bold) )
        self.info_label.setText("Params Used:\n        Reward Final State = "+str(rewards[0])+"        Reward Not Fianl State = "+str(rewards[1])+"\n        Episodies = "+str(ROUND)+ "        Convergence Threshold = "+str(convergence_threshold))
        self.info_label.setGeometry(int((20+445)*w_screen),SU+y+int(450*h_screen),int(560*w_screen),int(300))
        x+=int(135*w_screen)
        y+=int(20*h_screen)
        self.ag_label=QtWidgets.QLabel(self)
        self.ag_label.setGeometry(x,SU+int(y+(WIDTH+40*h_screen)*3-10),int(200*w_screen),WIDTH)
        self.ag_label.setText("Add AGENT")

        self.win_label=QtWidgets.QLabel(self)
        self.win_label.setGeometry(x,SU+y-int(WIDTH/2+40*h_screen)-20,int(200*w_screen),int(200*h_screen))
        self.win_label.setText("Add WIN")

        self.kit_label=QtWidgets.QLabel(self)
        self.kit_label.setGeometry(x,SU+int(y+(WIDTH+40*h_screen)*2),int(200*w_screen),WIDTH)
        self.kit_label.setText("Add KIT")


        self.ag2_label=QtWidgets.QLabel(self)
        self.ag2_label.setGeometry(x,SU+int(y+(WIDTH+40*h_screen)*5),int(200*w_screen),WIDTH)
        self.ag2_label.setText("Add AGENT 2")

        self.kit2_label=QtWidgets.QLabel(self)
        self.kit2_label.setGeometry(x,SU+int(y+(WIDTH+40*h_screen)*4)+10,int(200*w_screen),WIDTH)
        self.kit2_label.setText("Add KIT 2")

        self.wall_label=QtWidgets.QLabel(self)
        self.wall_label.setGeometry(x,SU+int(y+(WIDTH+40*h_screen)*1),int(200*w_screen),WIDTH)
        self.wall_label.setText("Add WALL")

        self.last_configuration=QtWidgets.QLabel(self)
        self.last_configuration.setGeometry(int(30*w_screen),SU+int(280*h_screen),int(300*w_screen),int(20*h_screen))
        self.last_configuration.setText("Preview last configuration: ")

        self.last=QtWidgets.QLabel(self)
       
        self.last.setGeometry(int(100*w_screen),SU+int(290*h_screen),int(300*w_screen),int(300*h_screen))

        self.labelVelocity=QtWidgets.QLabel("Test Speed",self)
        self.labelVelocity.setGeometry(int(40*w_screen),SU+int(620*h_screen+60*h_screen),int(150*w_screen),int(10*h_screen))
    
    #THREADS
        self.testTask = ShowTestMultiAgent()
        self.testTask.end.connect(self.goThreads)

        self.testTask2 = ShowTestSingleAgent()
        self.testTask2.end.connect(self.goThreads)
        self.testTask2.color.connect(self.colorTest)
        
        self.traintTask=Train()
        self.traintTask.dialog.connect(self.dialog)
        self.traintTask.end.connect(self.goThreads)

        self.t1=MyT(0)
        self.t1.change_value.connect(self.setProgressVal)
        self.t1.color.connect(self.colorSingle)
        self.t1.end.connect(self.endThread)

        self.t2=MyT(1)
        self.t2.color.connect(self.colorSingle)
        self.t2.change_value.connect(self.setProgressVal)
        self.t2.end.connect(self.endThread)

    #QSPINBOX AND FLAG
        self.ndet_level=QtWidgets.QSpinBox(self)
        self.ndet_level.setGeometry(int(250*w_screen),SU+int(176*h_screen),int(61*w_screen),int(31*h_screen))
        self.ndet_level.setMaximum(100)
        self.ndet_level.setMinimum(10)
        self.ndet_level.setValue(80)
        self.ndet_level.setSingleStep(10)
        self.ndet_level_label=QtWidgets.QLabel(self)
        self.ndet_level_label.setText("N_Det level")
        self.ndet_level_label.setGeometry(int((290+41)*w_screen),SU+int(186*h_screen),int(60*w_screen),int(20*h_screen))

        self.rows = QtWidgets.QSpinBox(self)
        self.rows.setGeometry(int(30*w_screen),SU+int(135*h_screen),int(51*w_screen),int(31*h_screen))
        self.rows.setValue(10)
        self.rows.setMaximum(30)
        
        self.label_rows=QtWidgets.QLabel(self)
        self.label_rows.setText("Environment's rows")
        self.label_rows.setGeometry(int((215-130)*w_screen),SU+int(145*h_screen),int(150*w_screen),int(20*h_screen))
        self.label_cols=QtWidgets.QLabel(self)
        self.label_cols.setText("Environment's cols")
        self.label_cols.setGeometry(int((215-130)*w_screen),SU+int((176+10)*h_screen),int(150*w_screen),int(20*h_screen))
        self.cols = QtWidgets.QSpinBox(self)
        self.cols.setValue(10)
        self.cols.setGeometry(int(30*w_screen),SU+int(176*h_screen),int(51*w_screen),int(31*h_screen))
        self.cols.setMaximum(30)
        
 
        self.velocity=QtWidgets.QSlider(parent=self,orientation=QtCore.Qt.Horizontal)
        self.velocity.setGeometry(int(30*w_screen+110*w_screen),SU+int(620*h_screen+50*h_screen),int(300*w_screen),int(40*h_screen))
        self.velocity.setRange(1,15)
        self.velocity.setValue(5)
        global VELOCITY
        VELOCITY=self.velocity
    
    #PRE_SHOW
        try :
            if platform.system() == 'Darwin' : 
                self.last.setPixmap(QPixmap(IMAGES_PATH+"screenshot.jpg"))
            else :
                self.last.setPixmap(QPixmap(IMAGES_PATH+"screenshot"))
        except :
            self.test.setEnabled(False)
            self.plot.setEnabled(False)
        global labelGrid
        for i in range(30):
            labelGrid.append([])
            for j in range(30):
                a=QLabel_Clicked(self,i,j)
                id=id+1
                labelGrid[i].append([a,"null"])
                a.setGeometry(0,0,0,0)
   
        self.kit2.setEnabled(False)
        self.kit.setEnabled(False)
        self.ag.setEnabled(False)
        self.win.setEnabled(False)
        self.wall.setEnabled(False)
        self.showMaximized()
        
    def showPlot(self):
        tmp = Plots_UI(self)
        # self.setCentralWidget(tmp)
        tmp.show()
           
    def newEnvironment(self):
        global COLS,ROWS,DETERMINISTICTEST,id,WIDTH,element,buttonclicked,AGENT,KIT,WIN,START, GRID,TERMINATED,LOSE_STATE
        self.ag2.setEnabled(False)
        self.kit2.setEnabled(False)
        LOSE_STATE =[]
        KIT=[0,0]
        START=[0,0]
        WIN = []
        self.newEnvironmentButton.setEnabled(True)
        AGENT=1
        element={"ag":False,"kit":False,"win":False,"ag2":False,"kit2":False,"wall":True}
        buttonclicked="null"
        ROWS=int(self.rows.value())
        COLS = int(self.cols.value())
        WIDTH=int((screen.height()-int(270*w_screen) -ROWS)/ROWS)
        DETERMINISTICTEST = True
        resetImageSize(WIDTH)
        GRID = []
       
        for i in range(30):
            if i < ROWS:
                GRID.append([])
            for j in range(30):
                if i < ROWS and j < COLS:
                    GRID[i].append(WHITE)
                    labelGrid[i][j][0].setGeometry(int(15*w_screen+int(50*w_screen+j*(WIDTH+1)+(screen.width()-screen.height()+int(270*w_screen))/2)),SU+int(40*h_screen+i*(WIDTH+1)+int(107*h_screen)),WIDTH,WIDTH)
                    labelGrid[i][j][0].setPixmap(whit)
                    labelGrid[i][j][1]="whit"
                    labelGrid[i][j][0].on.connect(self.mouseOnLabel)
                    labelGrid[i][j][0].installEventFilter(labelGrid[i][j][0])
                else:
                    labelGrid[i][j][1]="null"
                    labelGrid[i][j][0].clear()
                    labelGrid[i][j][0].setGeometry(0,0,0,0)
        self.kit.setEnabled(True)
        self.ag.setEnabled(True)
        self.win.setEnabled(True)
        self.wall.setEnabled(True)
        self.repaint()
    
    def initalize(self): 
        global TESTING,STOP, DETERMINISTICTEST
        global buttonclicked
        buttonclicked="wall"
        self.wall.setEnabled(True)
        self.trainButton.setEnabled(True)
        self.reward_button.setEnabled(True)
        self.choose_train_button.setEnabled(True)
        global ROWS,COLS,START,KIT,WIN,LOSE_STATE,WIDTH,TRAINING,GRID,AGENT,rewards
        env=np.load(CONFIGURATIONS_PATH+"environment.npy",allow_pickle=True)
        AGENT=env[0]
        ROWS=env[1]
        COLS=env[2]
        START=env[3]
        START[0]=tuple(START[0])
        WIN=env[4]
        KIT=env[5]
        KIT[0]=tuple(KIT[0])
        LOSE_STATE=env[6]
        rewards = env[7]
        ROUND = env[8]
        convergence_threshold = env[9]
        self.info_label.setText("Params Used:\n        Reward Final State = "+str(rewards[0])+"        Reward Not Fianl State = "+str(rewards[1])+"\n        Episodies = "+str(ROUND)+ "        Convergence Threshold = "+str(convergence_threshold))

        GRID = []
        for x in range(len(LOSE_STATE)):
            LOSE_STATE[x]=tuple(LOSE_STATE[x])
        if AGENT ==2:
            START[1]=tuple(START[1])
            KIT[1]=tuple(KIT[1])
        WIDTH=int((screen.height()-int(270*w_screen) -ROWS)/ROWS)
        resetImageSize(WIDTH)
        for i in range(ROWS):
            GRID.append([])
            for j in range(COLS):
                GRID[i].append(WHITE)
                labelGrid[i][j][0].setGeometry(int(15*w_screen+int(50*w_screen+j*(WIDTH+1)+(screen.width()-screen.height()+int(270*w_screen))/2)),SU+int(40*h_screen+i*(WIDTH+1)+int(107*h_screen)),WIDTH,WIDTH)
                labelGrid[i][j][0].setPixmap(whit)
                labelGrid[i][j][1]="whit"
                labelGrid[i][j][0].installEventFilter(labelGrid[i][j][0])
                labelGrid[i][j][0].on.connect(self.mouseOnLabel)
        labelGrid[START[0][0]][START[0][1]][0].setPixmap(ag)
        labelGrid[START[0][0]][START[0][1]][1]="ag"
        element["ag"]=True
        GRID[START[0][0]][START[0][1]] = GREEN
        labelGrid[KIT[0][0]][KIT[0][1]][0].setPixmap(kit)
        labelGrid[KIT[0][0]][KIT[0][1]][1]="kit"
        element["kit"] = True
        GRID[KIT[0][0]][KIT[0][1]] = BLUE
        labelGrid[WIN[0][0]][WIN[0][1]][0].setPixmap(win)
        labelGrid[WIN[0][0]][WIN[0][1]][1]=win
        GRID[WIN[0][0]][WIN[0][1]] = RED
        element["win"] = True
        if AGENT==2:

            labelGrid[START[1][0]][START[1][1]][0].setPixmap(ag2)
            labelGrid[KIT[1][0]][KIT[1][1]][0].setPixmap(kit2)
            labelGrid[START[1][0]][START[1][1]][1]="ag2"
            GRID[START[1][0]][START[1][1]] = GREEN
            labelGrid[KIT[1][0]][KIT[1][1]][1]="kit2"
            GRID[KIT[1][0]][KIT[1][1]] = BLUE
        for l in LOSE_STATE:
            labelGrid[l[0]][l[1]][0].setPixmap(wall)
            labelGrid[l[0]][l[1]][1]="wall"
            GRID[l[0]][l[1]] = BLACK
        self.repaint()
        if not element["ag2"]:
            self.ag2.setEnabled(True)
            self.kit2.setEnabled(True)
        self.trainButton.setEnabled(True)
        self.reward_button.setEnabled(True)
        self.choose_train_button.setEnabled(True)
        time.sleep(0.2)

    def testLast(self):
        global TESTING,STOP, DETERMINISTICTEST
        self.testTask2 = ShowTestSingleAgent()
        self.testTask = ShowTestMultiAgent()
        self.testTask2.end.connect(self.goThreads)
        self.testTask2.color.connect(self.colorTest)
        self.testTask.end.connect(self.goThreads)
        self.traintTask=Train()
        self.traintTask.dialog.connect(self.dialog)
        self.traintTask.end.connect(self.goThreads)
        DETERMINISTICTEST = True
        if not TESTING:
            self.catch=[False,False]
            STOP =False
            self.test.setText("Stop Test")
            TESTING=True
            self.allWidgets(False)
            self.velocity.setEnabled(True)
            self.wall.setEnabled(True)
            self.test.setEnabled(True)
            global buttonclicked
            buttonclicked="wall"
            global ROWS,COLS,START,KIT,WIN,LOSE_STATE,WIDTH,TRAINING,GRID,AGENT,rewards,ROUND,convergence_threshold
            TRAINING = False
            env=np.load(CONFIGURATIONS_PATH+"environment.npy",allow_pickle=True)
            AGENT=env[0]
            ROWS=env[1]
            COLS=env[2]
            START=env[3]
            START[0]=tuple(START[0])
            WIN=env[4]
            KIT=env[5]
            KIT[0]=tuple(KIT[0])
            LOSE_STATE=env[6]
            rewards = env[7]
            ROUND = env[8]
            convergence_threshold = env[9]
            self.info_label.setText("Params Used:\n        Reward Final State = "+str(rewards[0])+"        Reward Not Fianl State = "+str(rewards[1])+"\n        Episodies = "+str(ROUND)+ "        Convergence Threshold = "+str(convergence_threshold))
            GRID = []
            for x in range(len(LOSE_STATE)):
                LOSE_STATE[x]=tuple(LOSE_STATE[x])
            if AGENT ==2:
                START[1]=tuple(START[1])
                KIT[1]=tuple(KIT[1])
            WIDTH=int((screen.height()-int(270*w_screen) -ROWS)/ROWS)
            resetImageSize(WIDTH)
            for i in range(ROWS):
                GRID.append([])
                for j in range(COLS):
                    GRID[i].append(WHITE)
                    labelGrid[i][j][0].setGeometry(int(15*w_screen+int(50*w_screen+j*(WIDTH+1)+(screen.width()-screen.height()+int(270*w_screen))/2)),SU+int(40*h_screen+i*(WIDTH+1)+int(107*h_screen)),WIDTH,WIDTH)
                    labelGrid[i][j][0].setPixmap(whit)
                    labelGrid[i][j][1]="whit"
                    labelGrid[i][j][0].installEventFilter(labelGrid[i][j][0])
                    labelGrid[i][j][0].on.connect(self.mouseOnLabel)
            labelGrid[START[0][0]][START[0][1]][0].setPixmap(ag)
            labelGrid[START[0][0]][START[0][1]][1]="ag"
            GRID[START[0][0]][START[0][1]] = GREEN
            labelGrid[KIT[0][0]][KIT[0][1]][0].setPixmap(kit)
            labelGrid[KIT[0][0]][KIT[0][1]][1]="kit"
            GRID[KIT[0][0]][KIT[0][1]] = BLUE
            labelGrid[WIN[0][0]][WIN[0][1]][0].setPixmap(win)
            labelGrid[WIN[0][0]][WIN[0][1]][1]=win
            GRID[WIN[0][0]][WIN[0][1]] = RED
            if AGENT==2:
                labelGrid[START[1][0]][START[1][1]][0].setPixmap(ag2)
                labelGrid[KIT[1][0]][KIT[1][1]][0].setPixmap(kit2)
                labelGrid[START[1][0]][START[1][1]][1]="ag2"
                GRID[START[1][0]][START[1][1]] = GREEN
                labelGrid[KIT[1][0]][KIT[1][1]][1]="kit2"
                GRID[KIT[1][0]][KIT[1][1]] = BLUE
            for l in LOSE_STATE:
                labelGrid[l[0]][l[1]][0].setPixmap(wall)
                labelGrid[l[0]][l[1]][1]="wall"
                GRID[l[0]][l[1]] = BLACK
            self.repaint()
            time.sleep(0.2)
            if AGENT == 2:
                self.testTask2.start()
            else:
                self.testTask.start()
        else:
            STOP = True
            
            TESTING = False
 
    def train(self):
        global TRAINING,convergence_threshold
        TRAINING = True
        #convergence_threshold = float(self.convergence.text())
        self.screenShot()
        self.traintTask.start()
   
    def setProgressVal(self, val): 
        if STOP:
            self.progressBar.setValue(0)
            self.progressBar.close()
            self.setEnabled(True)
        else:
            self.progressBar.setValue(val)
            self.progressBar.repaint()
   
    def stopLearning(self):
        global STOP
        STOP=True
        self.trainButton.setEnabled(True)
        self.reward_button.setEnabled(True)
        self.choose_train_button.setEnabled(True)
   
    def mouseOnLabel(self,label):
        if buttonclicked=="wall":
            labelGrid[label[0]][label[1]][0].setPixmap(img[label[2]])
            labelGrid[label[0]][label[1]][1] = label[2]
            labelGrid[label[0]][label[1]][0].repaint()
   
    def winclick(self):
        global buttonclicked
        buttonclicked="win"
   
    def kitclick(self):
        global buttonclicked
        buttonclicked="kit"
   
    def agclick(self):
        global buttonclicked
        buttonclicked="ag"
   
    def ag2click(self):
        global buttonclicked
        buttonclicked="ag2"
   
    def wallclick(self):
        global buttonclicked
        buttonclicked="wall"
   
    def kit2click(self):
        global buttonclicked
        buttonclicked="kit2"        

class Rewards_UI(QtWidgets.QMdiSubWindow):
    end = pyqtSignal(list)
    def color_0(self):
        self.rewards[0].setStyleSheet("color: black")
    def color_1(self):
        self.rewards[1].setStyleSheet("color: black")
    def check(self):
        valid = True
        global rewards,w_screen,h_screen
        tmp = [0,0]
        try:
            tmp[0] = int(self.rewards[0].text())
            if len(self.rewards[0].text())==0:
                raise Exception
        except:
            valid = False
            self.rewards[0].setStyleSheet("color: red")
            if len(self.rewards[0].text())==0:
                self.rewards[0].setText("Add Value")
        try:
            tmp[1] = int(self.rewards[1].text())
            if len(self.rewards[1].text())==0:
                raise Exception
        except:
            valid = False
            self.rewards[1].setStyleSheet("color: red")
            if len(self.rewards[1].text())==0:
                self.rewards[1].setText("Add Value")
        if valid:
            rewards = tmp    
            self.end.emit([self])
    
    def __init__(self,parent=None):
        super(Rewards_UI,self).__init__(parent)
        self.setWindowTitle("Reward Setting")
        self.setGeometry(int(450*w_screen),int(220*h_screen),int(330*w_screen),int(230*h_screen))
        self.parent=parent
        self.labels = []
        self.rewards = []
        self.ok = QtWidgets.QPushButton(self)
        self.ok.setText("Done")
        self.ok.clicked.connect(self.check)
        self.ok.setGeometry(int(120*w_screen),int(120*h_screen),int(70*w_screen),int(40*h_screen))
        self.rewards.append(QtWidgets.QLineEdit(self))
        self.rewards.append(QtWidgets.QLineEdit(self))
        self.labels.append(QtWidgets.QLabel(self))
        self.labels.append(QtWidgets.QLabel(self))
        self.labels[0].setText("Reward Final State")
        self.labels[1].setText("Reward not Final State")
        self.labels[0].setGeometry(int(50*w_screen),int(40*h_screen),int(130*w_screen),int(30*h_screen))
        self.labels[1].setGeometry(int(50*w_screen),int(80*h_screen),int(180*w_screen),int(30*h_screen))
        self.rewards[0].setGeometry(int(250*w_screen),int(40*h_screen),int(60*w_screen),int(25*h_screen))
        self.rewards[1].setGeometry(int(250*w_screen),int(80*h_screen),int(60*w_screen),int(25*h_screen))
        self.rewards[0].setText(str(rewards[0]))
        self.rewards[1].setText(str(rewards[1]))
        self.rewards[0].textEdited.connect(self.color_0)
        self.rewards[1].textEdited.connect(self.color_1)

class Training_Settings_UI(QtWidgets.QMdiSubWindow):
    end = pyqtSignal(list)
    def check(self):
        valid = True
        global convergence_threshold,ROUND,w_screen,h_screen
        if self.checkbox.isChecked() :
            try:
                convergence_threshold = float(self.rewards[0].text())
                if len(self.rewards[0].text())== 0:
                    raise Exception
            except:
                valid = False
                self.rewards[0].setStyleSheet("color: red")
                if len(self.rewards[0].text())== 0:
                    self.rewards[0].setText("Add Value")
            try:
                ROUND = int(self.rewards[1].text())
                if len(self.rewards[1].text())== 0:
                    raise Exception
            except:
                valid = False
                self.rewards[1].setStyleSheet("color: red")
                if len(self.rewards[1].text())== 0:
                    self.rewards[1].setText("Add Value")
            
            
        else : 
            try:
                ROUND = int(self.rewards[1].text())
                convergence_threshold = -1
                if len(self.rewards[1].text())== 0:
                    raise Exception
            except:
                valid = False
                self.rewards[1].setStyleSheet("color: red")
                if len(self.rewards[1].text())== 0:
                     self.rewards[1].setText("Add Value")
        if valid:
            self.end.emit([self])
    def btnstate(self):
        if self.checkbox.isChecked() : 
            self.rewards[0].setEnabled(True)
            self.labels[0].setEnabled(True)
        else : 
            self.rewards[0].setEnabled(False)
            self.labels[0].setEnabled(False)
    def color_0(self):
        self.rewards[0].setStyleSheet("color: black")
    def color_1(self):
        self.rewards[1].setStyleSheet("color: black")
    def __init__(self,parent=None):
        super(Training_Settings_UI,self).__init__(parent)
        self.setWindowTitle("Train Setting")
        self.setGeometry(int(450*w_screen),int(220*h_screen),int(330*w_screen),int(230*h_screen))
        self.setFixedSize(int(330*w_screen),int(230*h_screen))
        self.parent=parent
        self.labels = []
        self.rewards = []
        self.ok = QtWidgets.QPushButton(self)
        self.checkbox = QtWidgets.QCheckBox(self)
        self.checkbox.setGeometry(int(25*w_screen),int(80*h_screen),int(180*w_screen),int(30*h_screen))
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.btnstate)
        self.ok.setText("Done")
        self.ok.clicked.connect(self.check)
        self.ok.setGeometry(int(120*w_screen),int(120*h_screen),int(70*w_screen),int(40*h_screen))
        self.rewards.append(QtWidgets.QLineEdit(self))
        self.rewards.append(QtWidgets.QLineEdit(self))
        self.labels.append(QtWidgets.QLabel(self))
        self.labels.append(QtWidgets.QLabel(self))
        self.labels[0].setText("Convergence Treshold")
        self.labels[1].setText("Max number of epochs")
        self.labels[0].setGeometry(int(50*w_screen),int(80*h_screen),int(180*w_screen),int(30*h_screen))
        self.labels[1].setGeometry(int(50*w_screen),int(40*h_screen),int(130*w_screen),int(30*h_screen))
        self.rewards[0].setGeometry(int(250*w_screen),int(80*h_screen),int(60*w_screen),int(25*h_screen))
        self.rewards[1].setGeometry(int(250*w_screen),int(40*h_screen),int(60*w_screen),int(25*h_screen))
        self.rewards[1].setText(str(ROUND))
        self.rewards[0].setText(str(convergence_threshold))
        self.rewards[0].textEdited.connect(self.color_0)
        self.rewards[1].textEdited.connect(self.color_1)

class Plots_UI(QtWidgets.QMdiSubWindow):

    def __init__(self,parent):
        super(Plots_UI,self).__init__(parent)
        self.title = 'Plots'
        self.left = 0
        self.top = 0
        self.width = 700
        self.height = 700
        self.setWindowTitle(self.title)
        self.setFixedSize(self.width, self.height)
        self.move(500,50)
        self.table_widget = TableWidget(self)
        self.table_widget.setGeometry(0,int(20),self.width,self.height-20)
    
class TableWidget(QtWidgets.QWidget):
    
    def __init__(self, parent):
        super(QtWidgets.QWidget, self).__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tabs.resize(400,400)
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Steps")
        self.tabs.addTab(self.tab2,"Reward")
        
        # Create first tab
        self.tab1.layout = QtWidgets.QVBoxLayout(self)
        self.tab1.setLayout(self.tab1.layout)
        self.tab2.layout = QtWidgets.QVBoxLayout(self)
        self.tab2.setLayout(self.tab2.layout)
        
        self.img1 = QtWidgets.QLabel(self)
        self.img2 = QtWidgets.QLabel(self)
        self.img1.setGeometry(0,0,self.img1.width(),self.img1.height())
        self.img2.setGeometry(0,0,self.img2.width(),self.img2.height())
        self.img1.setPixmap(QPixmap(IMAGES_PATH+'grafico1.png'))
        self.img2.setPixmap(QPixmap(IMAGES_PATH+'grafico2.png'))
        self.img1.repaint()
        self.img2.repaint()
        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.tab1.layout.addWidget(self.img1)
        self.tab2.layout.addWidget(self.img2)
        
    @pyqtSlot()
    def on_click(self):
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            currentQTableWidgetItem.repaint()

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()