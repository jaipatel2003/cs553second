import math
import gradio
from app import stop_inference, cancel_inference

def testsquare():
   num = 3
   assert 3*3 == 9

def test_cancel_inference():
   stop_inference = False
   cancel_inference()
   assert stop_inference == True
