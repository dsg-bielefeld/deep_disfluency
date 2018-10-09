# -*- coding: utf-8 -*-


class SpellingDict(dict):
    """Simple dict from regex -> normalized form.
    Reads from file in resource folder.
    """

    def __init__(self, path):
        f = open(path, 'r')
        for line in f:
            split = line.split('\t')
            self[split[0]] = split[1].strip("\n")
        f.close()
