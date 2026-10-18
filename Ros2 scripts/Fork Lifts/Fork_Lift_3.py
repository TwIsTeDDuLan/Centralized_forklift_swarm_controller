#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from forklift_controller import main

if __name__ == '__main__':
    main(forklift_id=3)
