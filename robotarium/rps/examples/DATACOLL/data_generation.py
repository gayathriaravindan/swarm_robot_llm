'''
Gayathri Aravindan
12/2024
'''
import random
import json
import numpy as np

def generate_data(num_samples):
    commands = {
        "circle": [
            "{speed} make a {size} circle",
            "{speed} form a {size} circular shape",
            "create a {size} circular formation {speed}",
            "arrange robots in a {size} circle {speed}",
            "{size} circular pattern",
            "robots should {speed} form a {size} circle",
            "organize into a {size} circle {speed}",
            "position in a {size} circular formation",
            "distribute in a {size} circle"
        ],
        "triangle": [
            "{speed} make a {size} triangle",
            "form a {size} triangular shape {speed}",
            "{speed} create a {size} triangle formation",
            "arrange in a {size} triangle",
            "position in a {size} triangle",
            "{speed} organize into a {size} triangle",
            "make a {size} three-sided shape",
            "form a {size} triangular pattern {speed}"
        ],
        "square": [
            "{speed} make a {size} square",
            "{speed} form a {size} square shape",
            "create a {size} square formation",
            "arrange in a {size} square",
            "four-sided {size} shape",
            "square {size} formation {speed}",
            "position in a {size} square",
            "{speed} organize into a {size} square",
            "{speed} form a {size} square pattern"
        ],
        "line": [
            "{speed} make a {size} line",
            "form a {size} straight line",
            "create a {size} linear formation",
            "arrange in a {size} line {speed}",
            "straight {size} formation {speed}",
            "linear {size} pattern",
            "{speed} position in a {size} line",
            "form a {size} linear shape"
        ],
        "leader": [
            "{speed} make the robots follow the leader along a {path} path",
            "guide the robots to follow the leader's movements in a {path} curve {speed}",
            "direct the robots to track a leader's {path} path and follow",
            "{speed} follow a leader making a {path} path",
            "make a leader do a {path} curve and the robots follow {speed}",
            "command the robots to follow the leader in a {path} path",
        ],
        "walk": [
            "make the robots move randomly in a {size} boundary",
            "have the robots perform a {size} random walk",
            "robots move unpredictably in {size} boundary",
            "follow random directions in {size} area",
            "explore environment in random steps in {size} place",
            "get the robots to move randomly in {size} walk"
        ],
        "consensus": [
            "meet at the center of the robots locations {speed}",
            "have the robots {speed} do a consensus",
            "{speed} align the robots at their center point",
            "form a group at the center {speed}",
            "do a consensus",
            "direct the robots to move towards a common center"
        ],
        "zigzag": [
            "make the robots do a zigzag",
            "move back and forth in a zigzag",
            "command the robots to traverse in a zigzag",
            "follow a sinusoidal zigzag",
            "direct the robots to follow a zigzag",
        ],
        "target": [
            "{speed} follow a target that is moving in a {path} path",
            "make the robots follow an object going in {path} trajectory {speed}",
            "command the swarm to pursue a target going in a {path} curve",
            "{speed} keep up with a target going in a {path} trajectory",
            "{speed} move toward a target in real-time going in a {path} path",
            "track an object going in a {path} curve"
        ],
        "disperse": [
            "{speed} make the robots spread out in a {size} boundary",
            "{speed} robots move away from each other in {size} boundary",
            "command the bots to position themselves further apart in a {size} area {speed}",
            "direct the robots to disperse in a {size} area",
            "guide the robots away from one another in a {size} place {speed}",
            "{speed} disperse with {size} size"
        ],
        "groups": [
            "{speed} divide into {number} groups",
            "split the robots into {number} different groups {speed}",
            "make the robots divide themselves into {number} groups",
            "have the robots split into {number} groups {speed}",
            "{speed} direct the robots to split into {number} subgroups",
            "get the robots to organize themselves into {number} subgroups {speed}"
        ]
    }

    sizes = {
        "small": [
            "small",
            "tiny",
            "short",
            "little"
        ],
        "medium": [
            "medium",
            "average"
        ],
        "large": [
            "large",
            "big",
            "gigantic",
            "huge",
            "long"
        ]
    }

    paths = [
        "sine", "cosine", "exponential", "parabola"
    ]

    speeds = {
        "slow": [
            "slow",
            "slowly",
            "gradually",
            "steadily"
        ],
        "fast": [
            "fast",
            "quickly",
            "swiftly",
            "rapid",
            "rapidly",
            "speedily"
        ]
    }

    numbers = [
        "two", "three", "four"
    ]

    data = []

    for _ in range(num_samples):
        function = random.choice(list(commands.keys()))
        command = random.choice(commands[function])
        speed_type = "slow"
        steps = 2000

        if "{speed}" in command:
            speed_type = random.choice(list(speeds.keys()))
            command = command.replace("{speed}", random.choice(speeds[speed_type]))
        if "{size}" in command:
            size_type = random.choice(list(sizes.keys()))
            command = command.replace("{size}", random.choice(sizes[size_type]))
        if "{path}" in command:
            path_type = random.choice(paths)
            command = command.replace("{path}", path_type)
        if "{number}" in command:
            num_type = random.choice(numbers)
            command = command.replace("{number}", num_type)
        print(function)
        print(command)

        if function == "circle":

            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)
    
            if size_type == "small":
                size = random.uniform(0.1, 0.7)
            elif size_type == "medium":
                size = random.uniform(0.7, 1.0)
            else:
                size = random.uniform(1.0, 2.0)
            
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
radius = {size}
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    x = r.get_poses()
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos_arr = np.array([radius * np.cos(theta), radius * np.sin(theta)])
    dxi = np.zeros((2, n))
    for i in range(n):
        error = pos_arr[:, i] - x[:2, i]
        dxi[:, i] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()
"""

        elif function == "triangle":
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)
            
            if size_type == "small":
                size = random.uniform(0.1, 0.3)
            elif size_type == "medium":
                size = random.uniform(0.3, 0.6)
            else:
                size = random.uniform(0.6, 1.0)
            
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
side_length = {size}
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    x = r.get_poses()
    vertices = np.array([
        [0, 0],
        [side_length, 0],
        [0.5 * side_length, np.sqrt(3) / 2 * side_length]
    ])
    pos_arr = []
    for i in range(3):
        start = vertices[i]
        end = vertices[(i + 1) % 3]
        num_points = n // 3 + (1 if i < n % 3 else 0)
        edge_points = np.linspace(start, end, num_points, endpoint=False)
        pos_arr.append(edge_points.T)
    pos_arr = np.hstack(pos_arr)[:, :n]
    dxi = np.zeros((2, n))
    for i in range(n):
        error = pos_arr[:, i] - x[:2, i]
        dxi[:, i] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()
"""

        elif function == "square":
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)
            
            if size_type == "small":
                size = random.uniform(0.1, 0.3)
            elif size_type == "medium":
                size = random.uniform(0.3, 0.6)
            else:
                size = random.uniform(0.6, 1.0)
            
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
side_length = {size}
formation_control_gain = {speed}
n = number_of_robots 
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    x = r.get_poses()
    vertices = np.array([
        [0, 0],
        [side_length, 0],
        [side_length, side_length],
        [0, side_length]
    ])
    pos_arr = []
    for i in range(4):
        start = vertices[i]
        end = vertices[(i + 1) % 4]
        num_points = n // 4 + (1 if i < n % 4 else 0)
        edge_points = np.linspace(start, end, num_points, endpoint=False)
        pos_arr.append(edge_points.T)
    pos_arr = np.hstack(pos_arr)[:, :n]
    dxi = np.zeros((2, n))
    for i in range(n):
        error = pos_arr[:, i] - x[:2, i]
        dxi[:, i] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()
"""

        elif function == "line":
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)
            
            if size_type == "small":
                size = random.uniform(0.1, 0.7)
            elif size_type == "medium":
                size = random.uniform(0.7, 1.0)
            else:
                size = random.uniform(1.0, 2.0)
            
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
length = {size}
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    x = r.get_poses()
    pos_arr = np.linspace([-length / 2, 0], [length / 2, 0], n).T
    dxi = np.zeros((2, n))
    for i in range(n):
        error = pos_arr[:, i] - x[:2, i]
        dxi[:, i] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()
"""
            
        elif function == "leader":
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)

            if path_type == "sine":
                leader_path = [np.array([t * 0.01, 0.5 * np.sin(2 * np.pi * t * 0.01)]) for t in range(2000)]
            elif path_type == "cosine":
                leader_path = [np.array([t * 0.01, 0.5 * np.cos(2 * np.pi * t * 0.01)]) for t in range(2000)]
            elif path_type == "exponential":
                leader_path = [np.array([t * 0.01, 0.1 * np.exp(0.1 * t * 0.01)]) for t in range(2000)]
            else:
                leader_path = [np.array([t * 0.01, 0.1 * (t * 0.01)**2]) for t in range(2000)]
        
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
leader_path = {leader_path}
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(min(steps, len(leader_path))):
    x = r.get_poses()
    leader_position = leader_path[step]
    pos_arr = np.zeros((2, n))
    pos_arr[:, 0] = leader_position
    for i in range(1, n):
        pos_arr[:, i] = x[:2, i - 1]
    dxi = np.zeros((2, n))
    for i in range(n):
        error = pos_arr[:, i] - x[:2, i]
        dxi[:, i] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()

"""

        elif function == "walk":
            if size_type == "small":
                size = random.uniform(0.1, 0.7)
            elif size_type == "medium":
                size = random.uniform(0.7, 1.0)
            else:
                size = random.uniform(1.0, 2.0)
            bounds = (-size, size, -size, size)
        
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
bounds = {bounds}
max_step_size = 1.0
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    x = r.get_poses()
    dxi = np.zeros((2, n))
    for i in range(n):
        dxi[:, i] = np.random.uniform(-max_step_size, max_step_size, 2)
        for j in range(2):
            if x[j, i] < bounds[j * 2] or x[j, i] > bounds[j * 2 + 1]:
                if x[j, i] < bounds[j * 2]:
                    dxi[j, i] += max_step_size * (bounds[j * 2] - x[j, i])
                elif x[j, i] > bounds[j * 2 + 1]:
                    dxi[j, i] += max_step_size * (bounds[j * 2 + 1] - x[j, i])
        dxi = si_barrier_cert(dxi, x[:2, :])
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()      
"""

        elif function == "consensus":
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    x = r.get_poses()
    center = np.mean(x[:2, :], axis=1, keepdims=True)
    dxi = np.zeros((2, n))
    for i in range(n):
        error = center[:, 0] - x[:2, i]
        dxi[:, i] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()
"""

        elif function == "zigzag":
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
amplitude = 0.2
frequency = 2.0
direction = (0.1,0)
n = number_of_robots
steps = {steps}      
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
t = 0
for step in range(steps):
    x = r.get_poses()
    dxi = np.zeros((2, n))
    for i in range(n):
        zig_zag_offset = amplitude * np.sin(frequency * t)
        desired_pos = x[:2, i] + np.array([direction[0], direction[1] + zig_zag_offset])
        error = desired_pos - x[:2, i]
        dxi[:, i] = 10 * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()
    t += 0.1      
"""
        elif function == "target":
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)
            
            if path_type == "sine":
                target_path = lambda t: np.array([t * 0.01, 0.5 * np.sin(2 * np.pi * t * 0.01)])
            elif path_type == "cosine":
                target_path = lambda t: np.array([t * 0.01, 0.5 * np.cos(2 * np.pi * t * 0.01)])
            elif path_type == "exponential":
                target_path = lambda t: np.array([t * 0.01, 0.1 * np.exp(0.1 * t * 0.01)])
            else:
                target_path = lambda t: np.array([t * 0.01, 0.1 * (t * 0.01)**2])
            
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
target_path = {target_path}
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    t = step / 100.0 
    x = r.get_poses()
    target_pos = target_path(t)
    dxi = np.zeros((2, n))
    for i in range(n):
        error = target_pos - x[:2, i]
        dxi[:, i] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()
"""

        elif function == "disperse":
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)

            if size_type == "small":
                size = random.uniform(0.1, 0.7)
            elif size_type == "medium":
                size = random.uniform(0.7, 1.0)
            else:
                size = random.uniform(1.0, 2.0)
            
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
boundary_radius = {size}
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
for step in range(steps):
    x = r.get_poses()
    dxi = np.zeros((2, n))
    for i in range(n):
        repulsion = np.zeros(2)
        for j in range(n):
            if i != j:
                distance = x[:2, i] - x[:2, j]
                if np.linalg.norm(distance) < boundary_radius / 2:
                    repulsion += distance / np.linalg.norm(distance)**2
        boundary_attraction = -x[:2, i] * (np.linalg.norm(x[:2, i]) - boundary_radius)
        dxi[:, i] = formation_control_gain * (repulsion + boundary_attraction)
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()            
"""

        else:
            if speed_type == "slow":
                speed = random.uniform(0.1, 0.5)
            else:
                speed = random.uniform(0.5, 1.0)
            
            if num_type == "two":
                group_centers = [[-0.5, -0.5], [0.5, 0.5]]
            elif num_type == "three":
                group_centers = [[-0.7, -0.7], [0, 0], [0.7, 0.7]]
            else:
                group_centers = [[-0.7, -0.7], [-0.7, 0.7], [0.7, -0.7], [0.7, 0.7]]
            
            code = f"""#
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

number_of_robots = 6
initial_conditions = np.zeros((3, number_of_robots))
theta_initial = np.linspace(0, 2 * np.pi, number_of_robots, endpoint=False)
initial_conditions[:2, :] = 0.1 * np.array([np.cos(theta_initial), np.sin(theta_initial)])
r = robotarium.Robotarium(
    number_of_robots=number_of_robots,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
    )
group_centers = {group_centers}
formation_control_gain = {speed}
n = number_of_robots
steps = {steps}
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
num_groups = len(group_centers)
group_assignments = np.array_split(np.arange(n), num_groups)
for step in range(steps):
    x = r.get_poses()
    dxi = np.zeros((2, n))
    for i, group in enumerate(group_assignments):
        group_center = np.array(group_centers[i])
        for robot in group:
            error = group_center - x[:2, robot]
            dxi[:, robot] = formation_control_gain * error
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_norm = (norms > 0.15)
    dxi[:, idxs_to_norm] *= 0.15 / norms[idxs_to_norm]
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(n), dxu)
    r.step()            
"""
        
        data_entry = {
            "input": command,
            "code": code
        }

        data.append(data_entry)
        
        with open("robotarium_samples.json", "w") as json_file:
            json.dump(data, json_file, indent=4)


generate_data(1000000)