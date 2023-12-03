import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_cylinder(ax, params):
    # Plotting the lateral surface of the cylinder
    x = np.linspace(0, params['length'], 30)
    theta = np.linspace(0, 2.*np.pi, 30)
    x, theta = np.meshgrid(x, theta)
    z = params['radius'] * np.cos(theta)
    y = params['radius'] * np.sin(theta)
    ax.plot_surface(x, y, z, color='#A1C3D1', alpha=1)
    
    # Plotting the front end of the cylinder
    theta = np.linspace(0, 2.*np.pi, 30)
    r = np.linspace(0, params['radius'], 30)
    theta, r = np.meshgrid(theta, r)
    x = np.zeros_like(theta)  # Make x a 2D array filled with zeros
    z = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot_surface(x, y, z, color='#A1C3D1', alpha=1)
    
    # Plotting the back end of the cylinder
    x = np.full_like(theta, params['length'])
    ax.plot_surface(x, y, z, color='#A1C3D1', alpha=1)

def plot_nose_cone(ax, params):
    phi = -np.linspace(0, 2 * np.pi, 100)
    x = -np.linspace(0, -params['nose_cone_height'], 30) - params['nose_cone_height']
    PHI, X = np.meshgrid(phi, x)
    
    nose_radius = params['radius'] * (1 - X/params['nose_cone_height'])**params['curvature_factor']
    
    Z = (X + params['nose_cone_height']) * np.cos(PHI) * nose_radius / params['nose_cone_height']
    Y = (X + params['nose_cone_height']) * np.sin(PHI) * nose_radius / params['nose_cone_height']
    
    ax.plot_surface(X, Y, Z, color='#D1A1C3', alpha=1)

def plot_flexible_fins(ax, params):
    A = [params['placement'], 0 + params['radius'], 0]
    B = [A[0] - params['base_length'], A[1], A[2]]
    C = [B[0] + params['front_shift'], A[1] + params['front_height'], A[2]]
    D = [A[0] - params['back_shift'], A[1] + params['back_height'], A[2]]
    
    normal = np.cross(np.array(B) - np.array(A), np.array(C) - np.array(A))
    normal = normal / np.linalg.norm(normal)
    
    vertices = []
    thickness_values = [params['thickness_A'], params['thickness_B'], params['thickness_C'], params['thickness_D']]
    
    for vertex, thickness in zip([A, B, C, D], thickness_values):
        vertex1 = np.array(vertex) + thickness * normal
        vertex2 = np.array(vertex) - thickness * normal
        vertices.append([vertex1.tolist(), vertex2.tolist()])
    
    faces = [
        [vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0]],
        [vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1]]
    ]
    
    for i in range(4):
        faces.append([vertices[i][0], vertices[i][1], vertices[(i + 1) % 4][1], vertices[(i + 1) % 4][0]])
    
    for face in faces:
        ax.add_collection3d(Poly3DCollection([face], facecolors='#D1A1C3', linewidths=1, edgecolors='#3A3335', alpha=.8))

    rotations = [90, 180, 270]
    for angle in rotations:
        rotated_faces = []
        angle_rad = np.radians(angle)
        for face in faces:
            rotated_face = []
            for vertex in face:
                x_rot = vertex[0]
                y_rot = vertex[1] * np.cos(angle_rad) - vertex[2] * np.sin(angle_rad)
                z_rot = vertex[1] * np.sin(angle_rad) + vertex[2] * np.cos(angle_rad)
                rotated_face.append([x_rot, y_rot, z_rot])
            rotated_faces.append(rotated_face)
        
        for face in rotated_faces:
            ax.add_collection3d(Poly3DCollection([face], facecolors='#D1A1C3', linewidths=1, edgecolors='#3A3335', alpha=.8))

def main():
    params = {
        'radius': 0.1,
        'length': 2.0,
        'placement': 1.9,
        'base_length': 0.5,
        'back_height': 0.15,
        'front_height': 0.1,
        'back_shift': 0.05,
        'front_shift': 0.1,
        'nose_cone_height': 0.75,
        'curvature_factor': 0.8,
        'thickness_A': 0.02,
        'thickness_B': 0.01,
        'thickness_C': 0.005,
        'thickness_D': 0.01
    }

    fig = plt.figure(figsize=(16, 12), facecolor='0.75')
    ax = fig.add_subplot(121, projection='3d', facecolor='0.75')
    plot_cylinder(ax, params)
    plot_flexible_fins(ax, params)
    plot_nose_cone(ax, params)

    padding_x = 0.1 * params['length']
    max_fin_height = max(params['back_height'], params['front_height'])
    padding_yz = 0.1 * params['radius']
    ax.set_xlim([-params['nose_cone_height'] - padding_x, params['length'] + padding_x])
    ax.set_ylim([-params['radius'] - max_fin_height - padding_yz, params['radius'] + max_fin_height + padding_yz])
    ax.set_zlim([-params['radius'] - max_fin_height - padding_yz, params['radius'] + max_fin_height + padding_yz])

    aspect_ratio = [params['nose_cone_height'] + params['length'] + 2 * padding_x,
                    2 * (params['radius'] + max_fin_height + padding_yz),
                    2 * (params['radius'] + max_fin_height + padding_yz)]
    ax.set_box_aspect(aspect_ratio)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
