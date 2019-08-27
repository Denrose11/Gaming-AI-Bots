import numpy as np
from alexnet import alexnet
WIDTH = 160 #resolution
HEIGHT = 120
LR = 1e-3# learning rate
EPOCHS = 10
MODEL_NAME = 'Developed_AI.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 22
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        Training_DATA = np.load('TRAINING_DATA.npy'.format(i))

        Training = Training_DATA[:-100]
        Testing = Training_DATA[-100:]

        A = np.array([i[0] for i in Training]).reshape(-1,WIDTH,HEIGHT,1)#feature sets for training
        B = [i[1] for i in Training] #labels or targets for training

        X_testing = np.array([i[0] for i in Testing]).reshape(-1,WIDTH,HEIGHT,1)#feature sets for testing
        Y_testing = [i[1] for i in Testing]#labels or targets for testing

        model.fit({'input': A}, {'targets': B}, n_epoch=1, validation_set=({'input': X_testing}, {'targets': Y_testing}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)









