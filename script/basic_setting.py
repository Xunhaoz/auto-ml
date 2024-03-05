import os
import warnings


def setting():
    if not os.path.exists('./file'):
        os.makedirs('./file')

    warnings.filterwarnings('ignore')
