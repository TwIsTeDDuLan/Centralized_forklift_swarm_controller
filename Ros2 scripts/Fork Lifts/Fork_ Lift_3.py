#!/usr/bin/env python3
import os
import sys

# Forward to Fork_Lift_3.py
script_dir = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(script_dir, "Fork_Lift_3.py")
if os.path.isfile(target):
    with open(target, "rb") as f:
        exec(compile(f.read(), target, "exec"))
