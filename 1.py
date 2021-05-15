from glob import glob
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

csv=pd.read_CSV('C:\\Users\\sns96\\Desktop\\통합 문서1.csv',names=['Column1','Column2'])
print(csv)