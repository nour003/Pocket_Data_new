import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

#################################
# Read strings in file
def read_strings(path):
    with open(path) as file:
        strings=[line.rstrip() for line in file]
    return strings

# Read floats
def read_floats(path):
    with open(path) as file:
        floats=[float(line.rstrip()) for line in file]
    return floats

