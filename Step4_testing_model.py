import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Developed_AI.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09

def go_just_straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def go_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)

def go_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)
    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            
            screen = grab_screen(region=(0,40,800,640))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            Prediction_of_AI = model.predict([screen.reshape(160,120,1)])[0]
            print(Prediction_of_AI)

            threshold_turning = .75
            threshold_moving_forward = 0.70

            if Prediction_of_AI[1] > threshold_moving_forward:
                go_just_straight()
            elif Prediction_of_AI[0] > threshold_turning:
                go_left()
            elif Prediction_of_AI[2] > threshold_turning:
                go_right()
            else:
                go_just_straight()

        keys = key_check()


        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       










