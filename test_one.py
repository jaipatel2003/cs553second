import math
from app import stop_inference, cancel_inference

def testsquare():
   num = 3
   assert 3*3 == 9

def test_cancel():
   global stop_inference
   stop_inference = False
   cancel_inference()
   assert stop_inference is True
