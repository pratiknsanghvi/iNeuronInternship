# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 20:42:46 2022

@author: pratiksanghvi
"""
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class InsPara(BaseModel):
    age: str
    bmi: str
    sex: str
    children: str
    smoker: str
    region: str