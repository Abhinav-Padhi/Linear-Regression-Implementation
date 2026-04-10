class Stochastic:
    def __init__(self,x,y,w,b,alpha=0.01,epochs=10000):
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.alpha = alpha
        self.epochs = epochs
    
    def update(self,standardized=True):
        n = len(self.x)
        pass
        