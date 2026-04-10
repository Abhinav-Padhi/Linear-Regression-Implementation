class MiniBatch:
    def __init__(self, x:list, y:list, batch_num, w, b, alpha=0.001, epochs=10000):
        self.x = x
        self.y = y
        self.batch_num = batch_num
        self.w = w
        self.b = b
        self.alpha = alpha 
        self.epochs = epochs
    
    def update(self,standardized=True):
        n = len(self.x)
        if standardized==True:
            mean_x = sum(xi for xi in self.x)/n 
            std_x = (sum((xi-mean_x)**2 for xi in self.x)/n)**0.5
            self.x = [(xi-mean_x)/std_x for xi in self.x]
            mean_y = sum(yi for yi in self.y)/n 
            std_y = (sum((yi-mean_y)**2 for yi in self.y)/n)**0.5
            self.y = [(yi-mean_y)/std_y for yi in self.y]
        if n%self.batch_num == 0:
            batch_size = n//self.batch_num
        else:
            batch_size = n//self.batch_num+1
        for i in range(self.batch_num):
            x_sum,y_sum,xy_sum,x2_sum=0,0,0,0
            x_batch = []
            y_batch = []
            if i == self.batch_num-1:
                x_batch = self.x[i*batch_size:]
                y_batch = self.y[i*batch_size:]
                batch_size = len(x_batch)
            else:
                x_batch = self.x[i*batch_size:(i+1)*batch_size]
                y_batch = self.y[i*batch_size:(i+1)*batch_size]
            for j in range(batch_size):
                x_sum+=x_batch[j]                            #summation of x
                y_sum+=y_batch[j]                            #summation of y
                x2_sum+=(x_batch[j]*x_batch[j])              #summation of x^2
                xy_sum+=(x_batch[j]*y_batch[j])              #summation of xy
            for _ in range(self.epochs):
                self.b,self.w = self.b - self.alpha*(-2/batch_size)*(y_sum - self.b*batch_size - self.w*x_sum),self.w - self.alpha*(-2/batch_size)*(xy_sum - self.b*x_sum - self.w*x2_sum)
        return [self.w,self.b]