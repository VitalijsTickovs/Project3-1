class Object:

    def __init__(self, name, probability):
        self.probability = probability
        self.name = name
        pass
    
    def getObjectName(self, name):
        return self.name
    
    def getObjectProbability(self, probability):
        return self.probability
    
    def setObjectProbability(self, probability):
        self.probability = probability
        return self.probability