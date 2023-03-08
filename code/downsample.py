#!/usr/bin/env python3

import util_tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys

def input_rate():
    try:
        downsample_rate = float(input('Input downsample rate(0-100): '))
        assert downsample_rate < 100
        return downsample_rate
    except ValueError:
        print('\tThat was no number')
        return input_rate()
    except AssertionError:
        print('\tInput must be between 0 and 100%')
        return input_rate()

downsample_rate = input_rate()

if len(sys.argv)==1:
    print('Error: No file was selected')

files = sys.argv[1:]
for file in files:
    print(f'Downsampling {file}...')
    output_name = 'downsize_'+file
    with open(file) as open_file:
        with open(output_name, 'w') as output:
            for i, line in enumerate(open_file):
                if np.random.randint(0, 100) < downsample_rate:
                    output.write(line)
    print('Done!')
