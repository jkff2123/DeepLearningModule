from Classes.Library import np

def loaddata(path, datatype):
    return np.loadtxt(path, dtype=datatype)

def savedata(path):
    return