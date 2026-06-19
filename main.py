# -*- coding: utf-8 -*-
"""
Character-Level Decoder Inference Pipeline

This repo demonstrates the inference pipeline for the trained
Character-Level Decoder-Only language model.
"""

from inference import generate
from model import MODEL

inp=input("Enter Some Starting word/Character: ")
generate(MODEL, inp, max_new_tokens=302)
print("\n")