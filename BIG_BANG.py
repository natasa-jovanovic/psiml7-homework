import numpy as np

fl = input().split()
N, S, T, P = int(fl[0]), int(fl[1]), int(fl[2]), float(fl[3])

particle = []
particle_temp = []
temp_dist = []
prob = []
for i in range(N):
    fl = input().split()
    particle.append([float(fl[0]), float(fl[1]), float(fl[2]), float(fl[3])])
    particle_temp.append([float(fl[0]), float(fl[1]), float(fl[2]), float(fl[3])])
    temp_dist.append(np.sqrt(particle[i][0] ** 2 + particle[i][1] ** 2))
    prob.append(1)

max = max(temp_dist)
Ta = 0
Tb = 0
particle = np.array(particle)
px, py, vx, vy = np.transpose(particle)
for i in range(10000):
    px -= vx
    py -= vy
    px[np.abs(px) > S] = np.sign(px[np.abs(px) > S]) * (2 * S - np.sign(px[np.abs(px) > S]) * px[np.abs(px) > S])
    vx[np.abs(px) > S] *= -1
    py[np.abs(py) > S] = np.sign(py[np.abs(py) > S]) * (2 * S - np.sign(py[np.abs(py) > S]) * py[np.abs(py) > S])
    vy[np.abs(py) > S] *= -1
    temp_dist = np.sqrt(px ** 2 + py ** 2)
    temp_max = np.max(temp_dist)

    if max - temp_max > 0.000001:
        max = temp_max
        Ta = i + 1

particle_temp = np.array(particle_temp)
particle_temp[:, 0:2] += T * particle_temp[:, 2:4]
Tb_temp = (particle_temp[:, 0:2] + S)/(2*S)
Tb_temp = np.sum(np.abs(np.floor(Tb_temp)), axis=1).astype(int)

Tb = sum(Tb_temp)
Tc = sum(P ** Tb_temp)

print(Ta, Tb, Tc)
