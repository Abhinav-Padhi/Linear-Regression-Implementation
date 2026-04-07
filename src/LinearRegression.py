class LinearRegression:
    def __init__(self, x:list, y:list, w, b, alpha=0.1, epochs=1000):
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.alpha = alpha
        self.epochs = epochs

    def update(self):
        x_sum,y_sum,xy_sum,x2_sum=0,0,0,0
        n = len(self.x)
        mean_x = sum(xi for xi in self.x)/n 
        std_x = (sum((xi-mean_x)**2 for xi in self.x)/n)**0.5
        self.x = [(xi-mean_x)/std_x for xi in self.x]
        mean_y = sum(yi for yi in self.y)/n 
        std_y = (sum((yi-mean_y)**2 for yi in self.y)/n)**0.5
        self.y = [(yi-mean_y)/std_y for yi in self.y]
        for i in range(len(self.x)):
            x_sum+=self.x[i]
            y_sum+=self.y[i]
            x2_sum+=(self.x[i]*self.x[i])
            xy_sum+=(self.x[i]*self.y[i])
        for _ in range(self.epochs):
            self.b,self.w = self.b - self.alpha*(-2/n)*(y_sum - self.b*n - self.w*x_sum),self.w - self.alpha*(-2/n)*(xy_sum - self.b*x_sum - self.w*x2_sum)
        return [self.b,self.w]

    