
import os
import numpy as np
import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy.stats import multivariate_normal

base_path = '/home/kswada/kw/robotics'


####################################################################################################
# --------------------------------------------------------------------------------------------------
# statistics basics
#  - sensor_data_200.txt
# --------------------------------------------------------------------------------------------------

data_path = os.path.join(base_path, '01_data', 'sensor_data', 'sensor_data_200.txt')

# data = pd.read_csv(data_path, delimiter="\t", header=None, names=("date", "time", "ir", "lidar"))
data = pd.read_csv(data_path, delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))

print(data)


# ----------
# histogram by matplotlib.pyplot
data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]), align='left')
plt.show()


# ----------
# mean value
mean1 = sum(data["lidar"].values)/len(data["lidar"].values)
mean2 = data["lidar"].mean()

print(f'{mean1} : {mean2}')


data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]), color="orange", align='left')
plt.vlines(mean1, ymin=0, ymax=5000, color="red")
plt.show()


# ----------
# sampling variance and unbiased variance
zs = data["lidar"].values
mean = sum(zs)/len(zs)
diff_square = [ (z - mean)**2 for z in zs]

sampling_var = sum(diff_square)/(len(zs))
unbiased_var = sum(diff_square)/(len(zs) - 1)

pandas_sampling_var = data["lidar"].var(ddof=False)
pandas_default_var = data["lidar"].var()

numpy_default_var = np.var(data["lidar"])
numpy_unbiased_var = np.var(data["lidar"], ddof=1)

# sampling variance
print(sampling_var)
print(pandas_sampling_var)
print(numpy_default_var)

# unbiased variance
# Pandas default is unbiased
print(unbiased_var)
print(pandas_default_var)
print(numpy_unbiased_var)


# ----------
# standard deviation
stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)
pandas_stddev = data["lidar"].std()

print(stddev1)
print(stddev2)
print(pandas_stddev)


# ----------
# probability distribution
freqs = pd.DataFrame(data["lidar"].value_counts())
freqs["probs"] = freqs["count"]/len(freqs["count"])

print(freqs)
print(freqs.transpose())

freqs["probs"].sort_index().plot.bar(color="blue")
plt.show()


# ----------
def drawing():
    return freqs.sample(n=1, weights="probs").index[0]

drawing()


samples = [drawing() for i in range(len(data))]
simulated = pd.DataFrame(samples, columns=["lidar"])
p = simulated["lidar"]
p.hist(bins=max(p) - min(p), color="orange", align='left')
plt.show()


# ----------
# normal distribution
def p(z, mu=209.7, dev=23.4):
    return math.exp(-(z - mu) ** 2 / (2 * dev)) / math.sqrt(2 * math.pi*dev)


zs = range(190, 230)
ys = [p(z) for z in zs]
plt.plot(zs, ys)
plt.show()


# scipy.stats.norm
zs = range(190, 230)
ys = [norm.pdf(z, mean1, stddev1) for z in zs]

plt.plot(zs, ys)
plt.show()

ys = [norm.cdf(z, mean1, stddev1) for z in zs]

plt.plot(zs, ys, color="red")
plt.show()


zs = range(190, 230)
ys = [norm.cdf(z+0.5, mean1, stddev1) - norm.cdf(z-0.5, mean1, stddev1) for z in zs]

plt.bar(zs, ys)
plt.show()


# ----------
# data ir
data["ir"].hist(bins=max(data["ir"]) - min(data["ir"]), align='left')
plt.show()

data.ir.plot()
plt.show()


# --------------------------------------------------------------------------------------------------
# statistics basics  (2-D gaussian)
#  - sensor_data_700.txt
# --------------------------------------------------------------------------------------------------

data_path = os.path.join(base_path, '01_data', 'sensor_data', 'sensor_data_700.txt')

data = pd.read_csv(data_path, delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))

print(data)


# ----------
d = data[(data["time"] < 160000) & (data["time"] >= 120000)]
d = d.loc[:, ["ir", "lidar"]]


# ----------
# IR and LIDAR joint distribution
sns.jointplot(d, x="ir", y="lidar", kind="kde")
plt.show()
plt.close()


# ----------
# variance
print(d.ir.var())
print(d.lidar.var())


# ----------
# covariance
diff_ir = d.ir - d.ir.mean()
diff_lidar = d.lidar - d.lidar.mean()
a = diff_ir * diff_lidar
print("covariance:", sum(a) / (len(d) - 1))

d.mean()

# covariance matrix
d.cov()


# ----------
# 2-D Gaussian
irlidar = multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)

x, y = np.mgrid[0:40, 710:750]
pos = np.empty(x.shape + (2,))
print(f'{x.shape} : {y.shape} : {pos.shape}')

pos[:, :, 0] = x
pos[:, :, 1] = y
cont = plt.contour(x, y, irlidar.pdf(pos))
cont.clabel(fmt='%1.1e')
plt.show()

# plt.close()


# ---------
# currently covariance is very small, almost zero.
# not co-variance is added 20
c = d.cov().values + np.array([[0, 20], [20, 0]])
print(c)

tmp = multivariate_normal(mean=d.mean().values.T, cov=c)
cont = plt.contour(x, y, tmp.pdf(pos))
plt.show()


# --------------------------------------------------------------------------------------------------
# statistics basics  (2-D gaussian)
#  - sensor_data_200.txt
# --------------------------------------------------------------------------------------------------

data_path = os.path.join(base_path, '01_data', 'sensor_data', 'sensor_data_200.txt')
data = pd.read_csv(data_path, delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))

print(data)


# ----------
d = data[(data["time"] < 160000) & (data["time"] >= 120000)]
d = d.loc[:, ["ir", "lidar"]]


# ----------
# IR and LIDAR joint distribution
sns.jointplot(d, x="ir", y="lidar", kind="kde")
plt.show()

plt.close()


# ----------
# this sensor does not fit to 2-D Gaussian, but try
irlidar = multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)

x, y = np.mgrid[280:340, 190:230]
pos = np.empty(x.shape + (2,))
print(f'{x.shape} : {y.shape} : {pos.shape}')

pos[:, :, 0] = x
pos[:, :, 1] = y
cont = plt.contour(x, y, irlidar.pdf(pos))
cont.clabel(fmt='%1.1e')
plt.show()

# plt.close()


# --------------------------------------------------------------------------------------------------
# statistics basics  (2-D gaussian)
#  - eigen value and eigen vector
# --------------------------------------------------------------------------------------------------

x, y = np.mgrid[0:200, 0:100]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

a = multivariate_normal(mean=[50, 50], cov=[[50, 0], [0, 100]])
b = multivariate_normal(mean=[100, 50], cov=[[125, 0], [0, 25]])
c = multivariate_normal(mean=[150, 50], cov=[[100, -25 * math.sqrt(3)], [-25 * math.sqrt(3), 50]])

for e in [a, b, c]:
    plt.contour(x, y, e.pdf(pos))

plt.gca().set_aspect('equal')
plt.gca().set_xlabel('x')
plt.gca().set_ylabel('y')

# plt.close()


# ----------
eig_vals, eig_vec = np.linalg.eig(c.cov)

print("eig_vals: ", eig_vals)
print("eig_vec: ", eig_vec)
print("eigen vector 1: ", eig_vec[:, 0])
print("eigen vector 2: ", eig_vec[:, 1])

print(np.array([[100, -25 * math.sqrt(3)], [-25 * math.sqrt(3), 50]]))
print(eig_vec @ np.diag(eig_vals) @ np.linalg.inv(eig_vec))


# ----------
plt.contour(x, y, c.pdf(pos))

# now vector length is short, multiple by 2
v = 2 * math.sqrt(eig_vals[0]) * eig_vec[:, 0]
plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color="red", angles='xy', scale_units='xy', scale=1)

v = 2 * math.sqrt(eig_vals[1]) * eig_vec[:, 1]
plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color="blue", angles='xy', scale_units='xy', scale=1)

plt.gca().set_aspect('equal')
plt.show()

plt.close()
