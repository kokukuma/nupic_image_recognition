#!/usr/bin/python
# coding: utf-8

def deepupdate(original, update):
    """
    Recursively update a dict.
    Subdict's won't be overwritten but also updated.
    """
    if update is None:
        return None
    for key, value in original.iteritems():
        if not key in update:
            update[key] = value
        elif update[key] == None:
            del update[key]
        elif isinstance(value, dict):
            deepupdate(value, update[key])
    return update

class DataBuffer(object):
    def __init__(self):
        self.stack = []

    def push(self, data):
        assert len(self.stack) == 0
        data = data.__class__(data)
        self.stack.append(data)

    def getNextRecordDict(self):
        assert len(self.stack) > 0
        return self.stack.pop()
