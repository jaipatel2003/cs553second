import math
from app import *

def testsquare():
   num = 3
   assert 3*3 == 9

def test_cancel_inference():
   global stop_inference
   stop_inference = False
   cancel_inference()
   assert stop_inference == True
