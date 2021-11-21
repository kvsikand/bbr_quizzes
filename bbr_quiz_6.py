import numpy as np
from scipy import optimize, linalg
from matplotlib import pyplot as plt

# Constants and constraints
q_s_min = 0
q_s_max = 3 * np.pi / 4
q_e_min = 0
q_e_max = 5 * np.pi / 6
q_h_min = 0
q_h_max = np.pi / 2

m_s = 1.93
m_e = 1.52
m_h = 0.52

I_s = 0.0141
I_e = 0.0188
I_h = 0.0003

dist_s = 0.165 
dist_e = 0.19
dist_h = 0.055

ideal_q = np.array([(q_s_min + q_s_max) / 2, (q_e_min + q_e_max) / 2, (q_h_min + q_h_max) / 2])

#limb lengths from table
l_s = 0.31
l_e = 0.34
l_h = 0.08

T = 1.0
movement_amplitude = 0.001

# Helper functions
# Get the jacobian for a given configuration
def get_jacobian(q):
  q_s, q_e, q_h = q
  # terms for jacobian
  s_s = np.sin(q_s)
  s_se = np.sin(q_s + q_e)
  s_seh = np.sin(q_s + q_e + q_h)
  c_s = np.cos(q_s)
  c_se = np.cos(q_s + q_e)
  c_seh = np.cos(q_s + q_e + q_h)
  # jacobian
  return np.array([
    [-(l_s*s_s + l_e*s_se + l_h*s_seh), -(l_e*s_se + l_h*s_seh), -l_h*s_seh],
    [l_s*c_s + l_e*c_se + l_h*c_seh, l_e*c_se + l_h*c_seh, l_h*c_seh]
  ])


# Get the target linear velocity of the endpoint at some timestep t
def get_velocity(t):
  t_n = t / T
  return (pow(t_n, 2) - 2 *t_n + 1) * (30 * movement_amplitude * pow(t_n, 2)) / T

# Get the position of the endpoint given a configuration
def get_pos(q):
  q_s, q_e, q_h = q
  s_s = np.sin(q_s)
  s_se = np.sin(q_s + q_e)
  s_seh = np.sin(q_s + q_e + q_h)
  c_s = np.cos(q_s)
  c_se = np.cos(q_s + q_e)
  c_seh = np.cos(q_s + q_e + q_h)
  x = l_s * c_s + l_e * c_se + l_h * c_seh
  y = l_s * s_s + l_e * s_se + l_h * s_seh
  return np.array([x, y])

# Initialization
q_initial = np.array([np.pi / 3, np.pi / 2, np.pi/2])
start_loc = get_pos(q_initial)
target_trajectory = [start_loc]

current_q_min_speed = q_initial.copy()
current_q_min_kinetic_energy = q_initial.copy()
qs_min_speed = [current_q_min_speed]
qs_min_kinetic_energy = [current_q_min_kinetic_energy]
loc = start_loc.copy()



for t in np.arange(0, T, 0.1):
  velocity_mag = get_velocity(t)
  x_dot = np.array([np.sqrt(velocity_mag / 2.), np.sqrt(velocity_mag / 2.)])
  # Compute the desired endpoint location at this timestep
  loc += x_dot
  target_trajectory.append(loc.copy())
  
  # Min joint speed version
  jacobian = get_jacobian(current_q_min_speed)
  # Equations 5.30 / 5.31 in the textbook
  # J_T = np.transpose(jacobian)
  # J_cross = J_T @ np.linalg.inv(jacobian @ J_T)
  J_cross = np.linalg.pinv(jacobian) # pseudoinverse. This function does the above two steps internally
  q_star = J_cross @ x_dot
  current_q_min_speed += q_star
  qs_min_speed.append(current_q_min_speed.copy())

  # Min kinetic energy version
  # define mass matrices
  # HOW TO DO THIS: (Equation 6.3, but that's just for a 2 joint system...)
  q_s, q_e, q_h = current_q_min_kinetic_energy
  H_11 = I_s + m_s * pow(dist_s, 2) + I_e + m_e * (pow(l_s, 2) + pow(dist_e, 2) + 2 *l_s * dist_e * np.cos(q_e)) + \
    I_h + m_h * (pow(l_s, 2) + pow(l_e, 2) + pow(dist_h, 2) + 2 *l_s * l_e * np.cos(q_e) + 2 *l_e * dist_h * np.cos(q_h) + 2 *l_s * dist_h * np.cos(q_e + q_h))
  
  H_12 = I_e + m_e * (pow(dist_e, 2) + l_s * dist_e * np.cos(q_e)) + I_h + \
    m_h * (pow(l_e, 2) + pow(dist_h, 2) + l_s * l_e * np.cos(q_e)) + 2 * l_e * dist_h * np.cos(q_h) + l_s * dist_h * np.cos(q_e + q_h)
  
  H_13 = I_h + m_h * (pow(dist_h, 2) + l_e * dist_h * np.cos(q_h)) + l_s * dist_h * np.cos(q_e + q_h)

  H_22 = I_e + m_e * pow(dist_e, 2) + I_h + m_h * (pow(l_e, 2) + pow(dist_h, 2) + 2 *l_e * dist_h * np.cos(q_h))

  H_23 = I_h + m_h * (pow(dist_h, 2) + l_e * dist_h * np.cos(q_h))

  H_33 = I_h + m_h * pow(dist_h, 2)

  H = np.zeros((3, 3))
  H[0, 0] = H_11
  H[0, 1] = H_12
  H[1, 0] = H_12
  H[1, 1] = H_22
  H[1, 2] = H_23
  H[2, 1] = H_23
  H[0, 2] = H_13
  H[2, 0] = H_13
  H[2, 2] = H_33

  J_H_cross = np.linalg.inv(H) @ jacobian.transpose() @ np.linalg.inv(jacobian @ np.linalg.inv(H) @ jacobian.transpose())
  q_h_star = J_H_cross @ x_dot
  current_q_min_kinetic_energy += q_h_star
  qs_min_kinetic_energy.append(current_q_min_kinetic_energy.copy())


target_trajectory = np.array(target_trajectory)


for config in qs_min_speed:
  q_s, q_e, q_h = config
  s_s = np.sin(q_s)
  s_se = np.sin(q_s + q_e)
  s_seh = np.sin(q_s + q_e + q_h)
  c_s = np.cos(q_s)
  c_se = np.cos(q_s + q_e)
  c_seh = np.cos(q_s + q_e + q_h)
  plt.gca().plot((0, l_s * c_s), (0, l_s * s_s), 'b-', lw=2)
  plt.gca().plot((l_s * c_s, l_s * c_s + l_e * c_se), (l_s * s_s, l_s * s_s + l_e * s_se), 'b-', lw=2)
  red_line, = plt.gca().plot((l_s * c_s + l_e * c_se, l_s * c_s + l_e * c_se + l_h * c_seh), (l_s * s_s + l_e * s_se, l_s * s_s + l_e * s_se + l_h * s_seh), 'b-', lw=2)

for config in qs_min_kinetic_energy:
  q_s, q_e, q_h = config
  s_s = np.sin(q_s)
  s_se = np.sin(q_s + q_e)
  s_seh = np.sin(q_s + q_e + q_h)
  c_s = np.cos(q_s)
  c_se = np.cos(q_s + q_e)
  c_seh = np.cos(q_s + q_e + q_h)
  plt.gca().plot((0, l_s * c_s), (0, l_s * s_s), 'g-', lw=2)
  plt.gca().plot((l_s * c_s, l_s * c_s + l_e * c_se), (l_s * s_s, l_s * s_s + l_e * s_se), 'g-', lw=2)
  green_line, = plt.gca().plot((l_s * c_s + l_e * c_se, l_s * c_s + l_e * c_se + l_h * c_seh), (l_s * s_s + l_e * s_se, l_s * s_s + l_e * s_se + l_h * s_seh), 'g-', lw=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend([red_line, green_line], ['Minimum joint velocity', 'Minimum kinetic energy'])

plt.savefig('fig_6c_6d.png')

