'''
Gayathri Aravindan
12/2024
'''

##imports
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

def move_in_circle(radius, formation_control_gain, r, n, steps):
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
    
def move_in_triangle(side_length, formation_control_gain, r, n, steps):
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
    
def move_in_square(side_length, formation_control_gain, r, n, steps):
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

def form_line(length, formation_control_gain, r, n, steps):
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

def follow_the_leader(leader_path, formation_control_gain, r, n, steps):
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

def random_walk(bounds, max_step_size, r, n, steps):
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

def consensus(formation_control_gain, r, n, steps):
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

def zig_zag_path(amplitude, frequency, direction, r, n, steps):
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

def target_following(target_path, formation_control_gain, r, n, steps):
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

def disperse(boundary_radius, formation_control_gain, r, n, steps):
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

def split_into_groups(group_centers, formation_control_gain, r, n, steps):
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

def main():
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
    bounds = (-0.5, 0.5, -0.5, 0.5)
    random_walk(bounds, 2.0, r, number_of_robots, 2000)
main()