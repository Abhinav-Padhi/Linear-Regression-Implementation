import LinearRegression
x = [1,2,3,4,5]
y = [3,5,7,9,11]
lr = LinearRegression.LinearRegression(x,y,-2,9,alpha = 0.01,epochs = 10000)
ans = lr.update()
print(f"{ans[0]:.4f}, {ans[1]:.4f}")
