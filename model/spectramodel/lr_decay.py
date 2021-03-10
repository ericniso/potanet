import math
from tensorflow.keras import callbacks

def exp_decay(initial_lrate, k):

    def decay(epoch):

        lrate = initial_lrate * math.exp(-k * epoch)
        return lrate
    
    return decay