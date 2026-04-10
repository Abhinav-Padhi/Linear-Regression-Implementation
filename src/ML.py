import GradientDescent
import MiniBatch
import pandas as pd
df = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
x = df['Experience Years']
y = df['Salary']
lr = GradientDescent.GradientDescent(x,y,-2,9,alpha = 0.01,epochs = 10000)
ans = lr.update(standardized=False)
print(f"{ans[0]:.4f}, {ans[1]:.4f}")

mb = MiniBatch.MiniBatch(x,y,1,0,0,alpha=0.001)
mb_ans = mb.update(standardized=True)

print(f"{mb_ans[0]:.4f}, {mb_ans[1]:.4f}")