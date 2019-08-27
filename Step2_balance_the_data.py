import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

Training_DATA = np.load('TRAINING_DATA.npy', allow_pickle=True)
print(len(Training_DATA))

df = pd.DataFrame(Training_DATA)
print(df.head()) # just to check the training data unbalancing
print(Counter(df[1].apply(str))) #counting the number of keys pressed for each key "W", "A","D"

Turn_Left = []
Turn_Right = []
Move_Forward = []

shuffle(Training_DATA) #we can shuffle it because the CNN doesnt tries to learn from previous frame
# and we need shuffling so that it doesnt not overfit to going forward
for data in Training_DATA:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        Turn_Left.append([img,choice])
    elif choice == [0,1,0]:
        Move_Forward.append([img,choice])
    elif choice == [0,0,1]:
        Turn_Right.append([img,choice])
    else:
        print('!!!!!!!!!!!!!')


Move_Forward = Move_Forward[:len(Turn_Left)][:len(Turn_Right)] # making the lengths of moving forward equal to turning left and right
Turn_Left = Turn_Left[:len(Move_Forward)]
Turn_Right = Turn_Right[:len(Move_Forward)]

final_data = Move_Forward + Turn_Left + Turn_Right
shuffle(final_data)
print(len(final_data))

np.save('TRAINING_DATA.npy', final_data)
