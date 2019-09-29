# -*- coding: utf-8 -*-
"""
Created on 2019/9/29 下午 03:22
@author: Ivan Y.W.Chiu
"""

# def printHello(func):
#     def wrapper():
#         print('Hello')
#         return func()
#     return wrapper
#
# @printHello
# def printWorld():
#     print('World')
#
# printWorld()

def printHello(func):
    def wrapper(arg):
        print('Hello')
        return func(arg)
    return wrapper

@printHello
def printArg(arg):
    print(arg)

printArg('World')
printArg('Kitty')