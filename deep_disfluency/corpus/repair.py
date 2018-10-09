# -*- coding: utf-8 -*-
class Repair:
    """
    Simple class which allows the construction of a repair
    object and stores the structure
    of the repair and other info.
    """

    def __init__(self, rnStart, iStart, RStart, REnd, repairID):
        assert rnStart < iStart, str(repairID) + str(rnStart) + str(iStart)
        assert iStart <= RStart, str(repairID) + str(iStart) + str(RStart)
        assert RStart <= REnd, str(repairID) + str(RStart) + str(REnd)
        self.repairID = repairID
        self.rnStart = rnStart
        self.iStart = iStart
        self.RStart = RStart
        self.REnd = REnd
        self.reparandumWords = []
        self.repairWords = []
        self.continuationWords = []
        self.caller = None  # need to set this
        self.reparandum = False  # True at the first word of reparandum
        self.repair = False  # True at first word of repair
        self.complete = False  # True after final word of repair

    def in_segment(self, pair):
        if pair < self.rnStart:
            return "o"
        elif pair < self.iStart:
            return "rm"
        elif pair < self.RStart:
            return "i"
        elif pair < self.REnd:
            return "rp"
        else:  # finished
            return None

    def classify(self):
        assert len(self.reparandumWords) > 0
        if len(self.repairWords) == 0:
            return "del"
        elif map(lambda x: x[0], self.repairWords) \
                == map(lambda x: x[0], self.reparandumWords):
            return "rep"
        else:
            return "sub"

    def to_string(self):
        string = str(self.repairID) + ":\n"
        string += str(self.rnStart) + "\n"
        string += str(self.iStart) + "\n"
        string += str(self.RStart) + "\n"
        string += str(self.REnd) + "\n"
        string += "caller = " + str(self.caller) + "\n"
        string += "reparandum =" + str(self.reparandum) + "\n"
        string += "repair = " + str(self.repair) + "\n"
        string += "complete = " + str(self.complete)
        return string

    def is_third_position(self):
        if self.REnd[0] - self.rnStart[0] > 1:
            return True
        return False
