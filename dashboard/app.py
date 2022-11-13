import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import percentile
from numpy.random import rand

from beats import display_beats
from aimodel import models

st.title("Assessment of Cardiac Dynamics")

st.sidebar.header("List of services:")
if st.sidebar.button("classification of beats"):
    display_beats()

if st.sidebar.button("Detection of Arrhthymia"):
    models()

