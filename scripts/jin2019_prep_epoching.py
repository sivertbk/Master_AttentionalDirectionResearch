import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from time import time
from autoreject import AutoReject, Ransac
import psutil
import gc
import json
from mne.preprocessing import ICA





process = psutil.Process()
rsc = Ransac()
ar = AutoReject()

montage = mne.channels.make_standard_montage('biosemi64')


