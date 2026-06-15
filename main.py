# -*- coding: utf-8 -*-
"""
Character-Level Decoder Inference Pipeline

This repo demonstrates the inference pipeline for the trained
Character-Level Decoder-Only language model.

Originally developed in Google Colab:
https://colab.research.google.com/drive/1CjvzQ7QYwtLuCXRpqcB29o5tAKjGgJrg
"""

from inference import generate
from model import MODEL

inp=input("Enter Some Starting word/Character: ")
generate(MODEL, inp, max_new_tokens=302)
print("\n")