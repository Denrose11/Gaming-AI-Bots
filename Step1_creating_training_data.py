import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os


def keys_pressed(keys): #keys pressed needs to be stored also converted these keys to boolean values
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


file_using = 'TRAINING_DATA.npy'

if os.path.isfile(file_using):
    print('File already exits, so loading previous file')
    data_used_for_training = list(np.load(file_using))
else:
    print('NO file exits, so making a new one')
    data_used_for_training = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):

        if not paused:

            screen = grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) #convert the colours to grayscale
            screen = cv2.resize(screen, (160,120))
            # getting the resized data which wil be the input to the CNN
            keys = key_check()
            output = keys_pressed(keys)
            data_used_for_training.append([screen,output])#appending constanly
            
            if len(data_used_for_training) % 1000 == 0: #saving traning data for every 100 frames
                print(len(data_used_for_training))
                np.save(file_using,data_used_for_training)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Continuing')
                time.sleep(1)
            else:
                print('Paused as directed by user')
                paused = True
                time.sleep(1)


main()


