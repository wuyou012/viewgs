import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
import math

def interpolate_translation(v0, v1, t):
    return (1 - t) * v0 + t * v1

def interpolate_rotation(R1, R2, r):
    """
    使用球面线性插值（SLERP）在两个旋转矩阵之间插值。

    参数:
    R1 (np.array): 第一个旋转矩阵，形状为 (3, 3)。
    R2 (np.array): 第二个旋转矩阵，形状为 (3, 3)。
    t (float): 插值参数，范围在 0 到 1 之间。

    返回:
    np.array: 插值后的旋转矩阵，形状为 (3, 3)。
    """
    # 将旋转矩阵转换为四元数
    Q1 = rot2quat(R1)
    Q2 = rot2quat(R2)

    # 使用 SLERP 进行插值
    Qt = slerp(Q1, Q2, r)

    # 将插值后的四元数转换回旋转矩阵
    return quat2rot(Qt)

def rot2quat(rotation_matrix):
    """
    TODO: Convert a rotation matrix to a quaternion.

    Parameters:
    rotation_matrix (np.array): A 3x3 or 4x4 NumPy array representing the rotation matrix.

    Returns:
    np.array: A NumPy array representing the quaternion.
    """
    # Ensure the input is a NumPy array
    rotation_matrix = np.array(rotation_matrix)

    # Check if the input is a 3x3 or 4x4 matrix
    if rotation_matrix.shape == (3, 3):
        # Convert the 3x3 rotation matrix to a quaternion
        r = R.from_matrix(rotation_matrix)
    elif rotation_matrix.shape == (4, 4):
        # Extract the top-left 3x3 submatrix if the input is a 4x4 matrix
        r = R.from_matrix(rotation_matrix[:3, :3])
    else:
        raise ValueError("The rotation matrix must be 3x3 or 4x4.")

    # Return the quaternion
    return r.as_quat()

def quat2rot(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    quaternion (np.array): A NumPy array representing the quaternion.

    Returns:
    np.array: A 3x3 NumPy array representing the rotation matrix.
    """
    # Ensure the input is a NumPy array
    quaternion = np.array(quaternion)

    # Check if the input is a valid quaternion
    if quaternion.shape[0] != 4:
        raise ValueError("The input must be a quaternion with four components.")

    # Convert the quaternion to a rotation object
    rotation = R.from_quat(quaternion)

    # Return the rotation matrix
    return rotation.as_matrix()

def slerp(q1, q2, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.

    Parameters:
    q1 (np.array): The first quaternion as a NumPy array.
    q2 (np.array): The second quaternion as a NumPy array.
    t (float): The interpolation parameter between 0 and 1.

    Returns:
    np.array: The interpolated quaternion.
    """
    # Convert the quaternions to scipy Rotation objects
    # r1 = R.from_quat(q1)
    # r2 = R.from_quat(q2)
    t = abs(t)
    if t > 1:
        t = 1

    key_rots = R.from_quat([q1, q2])
    key_times = [0, 1]

    slerp = Slerp(key_times, key_rots)
    times_to_interpolate = [t]

    interpolated_rotation = slerp(times_to_interpolate)[0]

    # Perform SLERP
    # interpolated_rotation = Slerp([0, 1], key_rots)

    # Return the interpolated quaternion
    return interpolated_rotation.as_quat()

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
