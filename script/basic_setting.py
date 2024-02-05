import os
import warnings


def setting():
    if not os.path.exists('file'):
        os.makedirs('file')

    if not os.path.exists('./temp'):
        os.makedirs('./temp')

    warnings.filterwarnings('ignore')
