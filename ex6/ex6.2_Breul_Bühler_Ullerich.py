#!/usr/bin/env python
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = pd.read_csv("swiss_roll.csv")

ax.scatter(df.x, df.y, df.z, c=df.color)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig("swiss_roll.png")
#plt.show()

iso = Isomap(n_components=2)
iso.fit(df.loc[:, df.columns != 'color'])
manifold_2Da = iso.transform(df.loc[:, df.columns != 'color'])
manifold_2D = pd.DataFrame(manifold_2Da, columns=['first', 'second'])
manifold_2D['color'] = df.color

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('2D Components from Isomap')
ax.set_xlabel('Component: 1')
ax.set_ylabel('Component: 2')

x_size = (max(manifold_2D['first']) - min(manifold_2D['first'])) * 0.08
y_size = (max(manifold_2D['second']) - min(manifold_2D['second'])) * 0.08

ax.scatter(manifold_2D['first'], manifold_2D['second'], c=manifold_2D['color'], marker='.', alpha=0.7)

plt.savefig("swiss_roll_result.png")
plt.show()

print(iso.explained_variance_)
