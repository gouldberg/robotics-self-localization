
import os
import numpy as np
import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

base_path = '/home/kswada/kw/robotics'


####################################################################################################
# --------------------------------------------------------------------------------------------------
# statistics basics:  Bayesian Estimation
#  - sensor_data_600.txt
# --------------------------------------------------------------------------------------------------
# sensor value when robot is located at 600 mm away from wall

data_path = os.path.join(base_path, '01_data', 'sensor_data', 'sensor_data_600.txt')

data = pd.read_csv(data_path, delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))


# ----------
# histogram
# data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]), align='left')
# plt.show()
# plt.close()


# ----------
# time series
# data.lidar.plot()
# plt.show()
# plt.close()

# data.ir.plot()
# plt.show()
# plt.close()


# ----------
# hourly mean
data["hour"] = [e//10000 for e in data.time]
d = data.groupby("hour")
# d.lidar.mean().plot()
# plt.show()
# plt.close()


# hour 6 and hour 14  (get_group)
# d.lidar.get_group(6).hist()
# d.lidar.get_group(14).hist()
# plt.show()
# plt.close()


# ----------
each_hour = {i : d.lidar.get_group(i).value_counts().sort_index() for i in range(24)}
print(each_hour[0])
print(each_hour[1])

# concatenate to columns
freqs = pd.concat(each_hour, axis=1)
freqs = freqs.fillna(0)
probs = freqs/len(data)
print(probs)

# sns.heatmap(probs)
# plt.show()
# plt.close()


# ----------
# joint distribution:  lidar * hour
# sns.jointplot(data, x="hour", y="lidar", kind="kde")
# plt.show()
# plt.close()


# ----------
# Bayesian Estimation

# by hour
p_t = pd.DataFrame(probs.sum())
# p_t.plot()
# p_t.transpose()
# plt.close()

# by value
p_z = pd.DataFrame(probs.transpose().sum()).sort_values("lidar")
# p_z.plot()
# p_z.transpose()
# plt.close()


# conditional probability distribution
cond_z_t = (probs/p_t[0]).sort_values("lidar")
print(cond_z_t)

# (cond_z_t[6]).plot.bar(color="blue", alpha=0.5)
# (cond_z_t[14]).plot.bar(color="orange", alpha=0.5)
# plt.show()


cond_t_z = probs.transpose()/probs.transpose().sum()

print("P(z=630) = ", p_z[0][630])
print("P(t=13) = ", p_t[0][13])
print("P(t=13 | z = 630) = ", cond_t_z[630][13])
print("Bayes P(z=630 | t = 13) = ", cond_t_z[630][13] * p_z[0][630] / p_t[0][13])
print("answer P(z=630 | t = 13) = ", cond_z_t[13][630])


# ----------
def bayes_estimation(sensor_value, current_estimation):
    new_estimation = []
    for i in range(24):
        new_estimation.append(cond_z_t[i][sensor_value] * current_estimation[i])
    return new_estimation / sum(new_estimation)


# when sensor value is 630, it is highly likely that the value is taken around 15:00
estimation = bayes_estimation(630, p_t[0])
print(estimation)
# plt.plot(estimation)
# plt.show()
# plt.close()


# sensor value [630, 632, 636] is taken around 5:00,
# this model estimate that it is highly likely that those values are taken around 3:00 - 8:00
values_5 = [630, 632, 636]
estimation = p_t[0]
for v in values_5:
    estimation = bayes_estimation(v, estimation)

print(estimation)
# plt.plot(estimation)
# plt.show()
# plt.close()


# sensor value [617, 624, 619] is taken around 11:00,
# this model estimate that it is highly likely that those values are taken around 13:00 - 15:00
values_11 = [617, 624, 619]
estimation = p_t[0]
for v in values_11:
    estimation = bayes_estimation(v, estimation)

print(estimation)
# plt.plot(estimation)
# plt.show()
# plt.close()

