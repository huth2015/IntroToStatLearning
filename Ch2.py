import pandas as pd
import numpy as np

# print function 
print('fit a model with', 11, 'variables ')

'hello' + ' ' + 'world'
print?

# Arrays
x = [3, 4, 5]
x

y = [4, 9, 7]
x+y

x = np.array([3, 4, 5])
y = np.array([4, 9, 7])

x + y

x = np.array([[1, 2], [3, 4]])
x
x.ndim
x.dtype

np.array?
np.array([[1, 2], [3, 4]], 'float64').dtype
np.array([[1, 2], [3, 4]], 'float64')
x.shape

x = np.array ([1, 2, 3, 4])
x.sum()
np.sum(x)

x = np.array ([1, 2, 3, 4, 5, 6])
print('beginning x:\n', x)
x_reshape = x.reshape ((2, 3))
print('reshaped x:\n', x_reshape)

x_reshape[0,0]
x_reshape[1,2]
x_reshape[2]

print('x before we modify x_reshape :\n', x)
print('x_reshape before we modify x_reshape :\n', x_reshape)
x_reshape [0, 0] = 5
print('x_reshape after we modify its top left element :\n',
x_reshape)
print('x after we modify top left element of x_reshape :\n', x)


x_reshape.shape , x_reshape.ndim , x_reshape.T
np.sqrt(x)
x**2
x**0.5

np.dot(x, x), x@x

# Using numpy's random number generator (normal distribution)
np.random.normal?

x = np.random.normal(size = 50)
x

y = x + np.random.normal(loc=50, scale=1, size =50)
y

np.corrcoef(x, y)

print(np.random.normal(scale=5, size =2))
print(np.random.normal(scale=5, size =2))

# Setting a seed with numpy to get the same result each time
rng = np.random.default_rng (1303)
print(rng.normal(scale=5, size =2))
rng2 = np.random.default_rng (1303)
print(rng2.normal(scale=5, size =2))

# Testing out mean, var, and standard deviation functions from numpy
rng = np.random.default_rng (3)
y = rng.standard_normal (10)
np.mean(y), y.mean ()

np.var(y), y.var(), np.mean((y - y.mean())**2)
np.sqrt(np.var(y)), np.std(y)

np.mean?
np.var?
np.std?

#Applying the functions above to a matrix
X = rng.standard_normal ((10, 3))
X

# mean of each column
X.mean(axis =0)

#mean of each row
X.mean(axis =1)

# Going through matplotlib
from matplotlib.pyplot import subplots
fig , ax = subplots(figsize =(8, 8))
x = rng.standard_normal (100)
y = rng.standard_normal (100)
ax.plot(x, y);

fig , ax = subplots(figsize =(8, 8))
ax.plot(x, y, 'o');

fig , ax = subplots(figsize =(8, 8))
ax.scatter(x, y, marker='o');

# titling the different axes
fig , ax = subplots(figsize =(8, 8))
ax.scatter(x, y, marker='o')
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y");

# changing the size in a plot that has already been created
fig.set_size_inches (12 ,3)
fig

#plotting multiple graphs in one pane
fig , axes = subplots(nrows=2,
ncols=3,
figsize =(15, 5))

axes [0 ,1]. plot(x, y, 'o')
axes [1 ,2]. scatter(x, y, marker='+')
fig

# Saving plots
fig.savefig("Figure.png", dpi =400)
fig.savefig("Figure.pdf", dpi =200);

axes [0 ,1]. set_xlim ([-1,1])
fig.savefig("Figure_updated.jpg")
fig

#Contour Plots
fig , ax = subplots(figsize =(8, 8))
x = np.linspace(-np.pi , np.pi , 50)
y = x
f = np.multiply.outer(np.cos(y), 1 / (1 + x**2))
ax.contour(x, y, f);

fig , ax = subplots(figsize =(8, 8))
ax.contour(x, y, f, levels =45);

?ax.contour

# Heat maps
fig , ax = subplots(figsize =(8, 8))
ax.imshow(f);