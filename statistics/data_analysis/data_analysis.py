import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

print(df)

p = []
v = []
for i in range(len(df)):
    x = df.loc[i,'graph']
    P = x[3:-9]
    new_P = P.replace("_","")
    V = x[6:-5]
    new_V = V.replace("_","")
    p.append(new_P)
    v.append(new_V)

df['Vértices'] = v
df['Prob'] = p
df.drop('Unnamed: 0', axis=1, inplace=True)


x  = df['Vértices']
plt.scatter(x, y = df['cuts'])
plt.xlabel('Vértices')
plt.ylabel('Cortes')
plt.title('Erdos-Renyi')
plt.show()