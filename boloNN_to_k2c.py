import sys
import os

# Add parent directory to path to import keras2c
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from keras2c import k2c

k2c('trained_network_keras.h5', 'boloNN', malloc=False, num_tests=10, verbose=True)