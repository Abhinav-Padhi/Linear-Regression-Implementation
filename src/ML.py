import LinearRegression
import MiniBatch
import pandas as pd
df = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
x = df['Experience Years']
y = df['Salary']
lr = LinearRegression.LinearRegression(x,y,-2,9,alpha = 0.01,epochs = 10000)
ans = lr.update()
print(f"{ans[0]:.4f}, {ans[1]:.4f}")

mb = MiniBatch.MiniBatch(x,y,8,0,0)
mb_ans = mb.update()
print(f"{mb_ans[0]:.4f}, {mb_ans[1]:.4f}")