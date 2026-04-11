import random
class Stochastic:
    def __init__(self,x,y,w,b,alpha=0.001,epochs=1000):
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.alpha = alpha
        self.epochs = epochs
    
    def update(self,standardized=True):
        n = len(self.x)
        if standardized==True:
            mean_x = sum(xi for xi in self.x)/n                        #mean of x
            std_x = (sum((xi-mean_x)**2 for xi in self.x)/n)**0.5      #standard deviation of x
            self.x = [(xi-mean_x)/std_x for xi in self.x]              #normalizing x
            mean_y = sum(yi for yi in self.y)/n                        #mean of y
            std_y = (sum((yi-mean_y)**2 for yi in self.y)/n)**0.5      #standard deviation of y
            self.y = [(yi-mean_y)/std_y for yi in self.y]              #normalizing y
        error = []
        for i in range(self.epochs):
            data = list(zip(self.x,self.y))
            random.shuffle(data)
            for X,Y in data:
                e = -Y + self.b + self.w*X
                self.b = self.b - 2*self.alpha*e 
                self.w = self.w - 2*self.alpha*e*X
        return [self.w,self.b]
        