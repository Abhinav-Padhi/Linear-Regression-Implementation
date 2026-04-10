class GradientDescent:
    def __init__(self, x:list, y:list, w, b, alpha=0.1, epochs=1000):
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.alpha = alpha
        self.epochs = epochs

    def update(self,standardized = True):
        x_sum,y_sum,xy_sum,x2_sum=0,0,0,0
        n = len(self.x)
        if standardized==True:
            mean_x = sum(xi for xi in self.x)/n                        #mean of x
            std_x = (sum((xi-mean_x)**2 for xi in self.x)/n)**0.5      #standard deviation of x
            self.x = [(xi-mean_x)/std_x for xi in self.x]              #normalizing x
            mean_y = sum(yi for yi in self.y)/n                        #mean of y
            std_y = (sum((yi-mean_y)**2 for yi in self.y)/n)**0.5      #standard deviation of y
            self.y = [(yi-mean_y)/std_y for yi in self.y]              #normalizing y
        for i in range(len(self.x)):
            x_sum+=self.x[i]                            #summation of x
            y_sum+=self.y[i]                            #summation of y
            x2_sum+=(self.x[i]*self.x[i])               #summation of x^2
            xy_sum+=(self.x[i]*self.y[i])               #summation of xy
        for _ in range(self.epochs):
            self.b,self.w = self.b - self.alpha*(-2/n)*(y_sum - self.b*n - self.w*x_sum),self.w - self.alpha*(-2/n)*(xy_sum - self.b*x_sum - self.w*x2_sum)   #weight and bias updates
        return [self.w,self.b]

