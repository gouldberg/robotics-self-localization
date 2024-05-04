
import os
import numpy as np
import math
import copy

import matplotlib
# matplotlib.use('nbagg')

import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import expon, norm, uniform

base_path = '/home/kswada/kw/robotics'


# --------------------------------------------------------------------------------------------------
# simple robot model
# --------------------------------------------------------------------------------------------------

class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval
    # ----------
    def append(self, obj):
        self.objects.append(obj)
    # ----------
    def draw(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        elems = []
        if self.debug:
            for i in range(int(self.time_span / self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(
                fig, self.one_step, fargs=(elems, ax),
                frames=int(self.time_span / self.time_interval) + 1,
                interval=int(self.time_interval * 1000), repeat=False)
            plt.show()
    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)


class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose = pose
        self.r = 0.2  # radius
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor
    # ----------
    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)
    # ----------
    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array(
                [
                    nu * math.cos(t0),
                    nu * math.sin(t0),
                    omega
                ]) * time
        else:
            return pose + np.array(
                [
                    nu / omega * (math.sin(t0 + omega * time) - math.sin(t0)),
                    nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)),
                    omega * time
                ])
    # ----------
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        if self.sensor: self.sensor.data(self.pose)


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
    # ----------
    def decision(self, observation=None):
        return self.nu, self.omega


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None
    # ----------
    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


class Map:
    def __init__(self):
        self.landmarks = []
    # ----------
    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)
    # ----------
    def draw(self, ax, elems):
        for lm in self.landmarks: lm.draw(ax, elems)


class IdealCamera:
    def __init__(self, env_map,
                 distance_range=(0.5, 6.0),
                 direction_range=(-math.pi / 3, math.pi / 3)):
        self.map = env_map
        self.lastdata = []
        self.distance_range = distance_range
        self.direction_range = direction_range
    # ----------
    def visible(self, polarpos):
        if polarpos is None:
            return False
        return (self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
                and self.direction_range[0] <= polarpos[1] <= self.direction_range[1])
    # ----------
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))
        # ----------
        self.lastdata = observed
        return observed
    # ----------
    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2 * np.pi
        while phi < -np.pi: phi += 2 * np.pi
        return np.array([np.hypot(*diff), phi]).T
    # ----------
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color="pink")


# --------------------------------------------------------------------------------------------------
# Robot and Camera
# --------------------------------------------------------------------------------------------------

class Robot(IdealRobot):
    # ----------
    def __init__(self, pose, agent=None, sensor=None, color="black",
                 noise_per_meter=5, noise_std=math.pi / 60, bias_rate_stds=(0.1, 0.1),
                 expected_stuck_time=1e100, expected_escape_time=1e-100,
                 expected_kidnap_time=1e100, kidnap_range_x=(-5.0, 5.0), kidnap_range_y=(-5.0, 5.0)):
        super().__init__(pose, agent, sensor, color)
        self.noise_pdf = expon(scale=1.0 / (1e-100 + noise_per_meter))
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale=noise_std)
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])
        # ----------
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.is_stuck = False
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        # ----------
        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0), scale=(rx[1] - rx[0], ry[1] - ry[0], 2 * math.pi))
    # ----------
    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu) * time_interval + self.r * abs(omega) * time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()
        return pose
    # ----------
    def bias(self, nu, omega):
        return nu * self.bias_rate_nu, omega * self.bias_rate_omega
    # ----------
    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True
        return nu * (not self.is_stuck), omega * (not self.is_stuck)
    # ----------
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose
    # ----------
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)


# ----------
class Camera(IdealCamera):
    def __init__(self, env_map,
                 distance_range=(0.5, 6.0),
                 direction_range=(-math.pi / 3, math.pi / 3),
                 distance_noise_rate=0.1, direction_noise=math.pi / 90,
                 distance_bias_rate_stddev=0.1, direction_bias_stddev=math.pi / 90,
                 phantom_prob=0.0, phantom_range_x=(-5.0, 5.0), phantom_range_y=(-5.0, 5.0),
                 oversight_prob=0.1, occlusion_prob=0.0):
        super().__init__(env_map, distance_range, direction_range)
        # ----------
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale=direction_bias_stddev)
        # ----------
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale=(rx[1] - rx[0], ry[1] - ry[0]))
        self.phantom_prob = phantom_prob
        # ----------
        self.oversight_prob = oversight_prob
        self.occlusion_prob = occlusion_prob
    # ----------
    def noise(self, relpos):
        ell = norm.rvs(loc=relpos[0], scale=relpos[0] * self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T
    # ----------
    def bias(self, relpos):
        return relpos + np.array([relpos[0] * self.distance_bias_rate_std,
                                  self.direction_bias]).T
    # ----------
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos)
        else:
            return relpos
    # ----------
    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos
    # ----------
    def occlusion(self, relpos):
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0] + uniform.rvs() * (self.distance_range[1] - relpos[0])
            return np.array([ell, relpos[1]]).T
        else:
            return relpos
    # ----------
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z)
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z, lm.id))
        self.lastdata = observed
        return observed


####################################################################################################
# --------------------------------------------------------------------------------------------------
# noise simulation - 1
# --------------------------------------------------------------------------------------------------

circling = Agent(0.2, 10.0 / 180 * math.pi)

r = Robot(pose=np.array([0, 0, 0]).T, sensor=None, agent=circling, color="gray")


# ----------
world = World(30, 0.1)

for i in range(100):
    circling = Agent(0.2, 10.0 / 180 * math.pi)
    r = Robot(pose=np.array([0, 0, 0]).T, sensor=None, agent=circling, color="gray")
    world.append(r)


world.draw()

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 2
#  - nu and omega bias
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1)

circling = Agent(0.2, 10.0 / 180 * math.pi)


# ----------
# no noise, no bias
nobias_robot = IdealRobot(pose=np.array([0, 0, 0]).T, sensor=None, agent=circling, color="gray")
world.append(nobias_robot)


# ----------
# no noise, but with bias
biased_robot = Robot(pose=np.array([0, 0, 0]).T, sensor=None, agent=circling, color="red",
                     noise_per_meter=0, bias_rate_stds=(0.2, 0.2))
world.append(biased_robot)


world.draw()


# ----------
save_ani_path = os.path.join(base_path, '04_output', 'anm.gif')
world.ani.save(save_ani_path, writer='pillow', fps=10)

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 3
#  - stuck
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1)

circling = Agent(0.2, 10.0 / 180 * math.pi)


# expected_stuck_time = 60.0
# expected_escape_time = 60.0
expected_stuck_time = 2.0
expected_escape_time = 2.0

# 10 robots with stucking
for i in range(10):
    r = Robot(
        pose=np.array([0, 0, 0]).T,
        sensor=None, agent=circling, color="gray", noise_per_meter=0, bias_rate_stds=(0.0, 0.0),
        expected_stuck_time=expected_stuck_time, expected_escape_time=expected_escape_time)
    world.append(r)


# no stucking
r = IdealRobot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="red")
world.append(r)


# ----------
world.draw()

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 4
#  - kid-napping
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1)

circling = Agent(0.2, 10.0 / 180 * math.pi)

for i in range(2):
    r = Robot(
        np.array([0, 0, 0]).T,
        sensor=None, agent=circling, color="gray",
        noise_per_meter=0, bias_rate_stds=(0.0, 0.0), expected_kidnap_time=5)
    world.append(r)


# ----------
r = IdealRobot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="red")
world.append(r)


# ----------
world.draw()

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 6 / 7
#  - sensor with noise
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1)


# map and landmark
m = Map()
m.append_landmark(Landmark(-4, 2))
m.append_landmark(Landmark(2, -3))
m.append_landmark(Landmark(3, 3))
world.append(m)

circling = Agent(0.2, 10.0 / 180 * math.pi)

# in default setting, camera has noise
r = Robot(np.array([0, 0, 0]).T, sensor=Camera(m), agent=circling)
# r = Robot(np.array([0, 0, math.pi / 6]).T, sensor=Camera(m), agent=circling)

world.append(r)

world.draw()

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 8
#  - sensor with bias
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1)

m = Map()
m.append_landmark(Landmark(-4, 2))
m.append_landmark(Landmark(3, -3))
m.append_landmark(Landmark(3, 3))
m.append_landmark(Landmark(3, -2))
m.append_landmark(Landmark(3, 0))
m.append_landmark(Landmark(3, 1))
world.append(m)

straight = Agent(0.2, 0.0)

# in default setting, camera has also bias
r = Robot(np.array([0, 0, 0]).T, sensor=Camera(m), agent=straight)
world.append(r)


# ----------
world.draw()

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 9
#  - phantom
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1)

m = Map()
m.append_landmark(Landmark(-4, 2))
m.append_landmark(Landmark(2, -3))
m.append_landmark(Landmark(3, 3))
world.append(m)

straight = Agent(0.2, 0.0)
circling = Agent(0.2, 10.0/180*math.pi)

# set phantom prob
r = Robot(np.array([0, 0, math.pi/6]).T, sensor=Camera(m, phantom_prob=0.5), agent=circling)
world.append(r)


# ----------
world.draw()

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 10
#  - oversight
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1)


m = Map()
m.append_landmark(Landmark(-4, 2))
m.append_landmark(Landmark(2, -3))
m.append_landmark(Landmark(3, 3))
world.append(m)


straight = Agent(0.2, 0.0)
circling = Agent(0.2, 10.0 / 180 * math.pi)

# set oversight prob
r = Robot(np.array([0, 0, math.pi / 6]).T, sensor=Camera(m, oversight_prob=0.1), agent=circling)
world.append(r)


world.draw()

# plt.close()


# --------------------------------------------------------------------------------------------------
# noise simulation - 11
#  - occlusion
# --------------------------------------------------------------------------------------------------

world = World(30, 0.1, debug=False)

m = Map()
m.append_landmark(Landmark(-4, 2))
m.append_landmark(Landmark(2, -3))
m.append_landmark(Landmark(3, 3))
world.append(m)

straight = Agent(0.2, 0.0)
circling = Agent(0.2, 10.0 / 180 * math.pi)

# set occlusion prob
r = Robot(np.array([2, 2, math.pi / 6]).T, sensor=Camera(m, occlusion_prob=0.1), agent=circling)
world.append(r)


# ----------
world.draw()

# plt.close()
