# Deep learning framework: Number of weights = no of feautures

# ANN: Each node has its own weights, biases, and activation function

# Typical node count for hidden layer: 2^n

# Import dependencies

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


