#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct
import json

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                if xyzs is None:
                    xyzs = xyz[None, ...]
                    rgbs = rgb[None, ...]
                    errors = error[None, ...]
                else:
                    xyzs = np.append(xyzs, xyz[None, ...], axis=0)
                    rgbs = np.append(rgbs, rgb[None, ...], axis=0)
                    errors = np.append(errors, error[None, ...], axis=0)
    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def read_camera_params(file_path):
    camera_params = {
        "R": [],
        "T": [],
        "focals": None,
        "H": None,
        "W": None,
        "images": []
    }
    with open(file_path, 'r') as f:
        lines = f.readlines()
        image_names = []
        image_paths = []
        for line in lines:
            if line.startswith("R:"):
                camera_params["R"].append(json.loads(line[3:]))
            elif line.startswith("T:"):
                camera_params["T"].append(json.loads(line[3:]))
            elif line.startswith("focals:"):
                camera_params["focals"] = json.loads(line[8:])
            elif line.startswith("H:"):
                camera_params["H"] = int(float(line[3:].strip()))
            elif line.startswith("W:"):
                camera_params["W"] = int(float(line[3:].strip()))
            elif line.startswith("image_name:"):
                image_name = line[12:].strip()
                image_names.append(image_name)
            elif line.startswith("image_path:"):
                image_path = line[12:].strip()
                image_paths.append(image_path)
        for i in range(len(image_names)):
            camera_params["images"].append({"name": image_names[i],
                                            "path": image_paths[i]})
    return camera_params

def split_camera_params_old(camera_params):
    mid_camera_params = {
        "R": camera_params["R"][1:-1],
        "T": camera_params["T"][1:-1],
        "focals": camera_params["focals"],
        "H": camera_params["H"],
        "W": camera_params["W"]
    }
    real_camera_params = {
        "R": camera_params["R"][:1] + camera_params["R"][-1:],
        "T": camera_params["T"][:1] + camera_params["T"][-1:],
        "focals": camera_params["focals"],
        "H": camera_params["H"],
        "W": camera_params["W"],
        "images": camera_params["images"]
    }
    return real_camera_params, mid_camera_params

def split_camera_params_mid(camera_params):
    """
    将相机参数分为关键帧(real_camera_params)和中间帧(mid_camera_params)。
    关键帧为第1, 25, 49帧和所有满足24*k+1的帧。

    Args:
        camera_params: 包含相机参数的字典，格式为:
            {
                "R": [r1, r2, ..., rn],         # 旋转矩阵列表
                "T": [t1, t2, ..., tn],         # 平移向量列表
                "focals": [fx, fy],             # 焦距
                "H": height,                    # 图像高度
                "W": width,                     # 图像宽度
                "images": [                     # 图像信息
                    {"name": "img1.jpg", "path": "/path/to/img1.jpg"},
                    ...
                ]
            }

    Returns:
        tuple: (real_camera_params, mid_camera_params)
    """
    real_camera_params = {
        "R": [],
        "T": [],
        "focals": camera_params["focals"],
        "H": camera_params["H"],
        "W": camera_params["W"],
        "images": camera_params["images"]
    }
    mid_camera_params = {
        "R": [],
        "T": [],
        "focals": camera_params["focals"],
        "H": camera_params["H"],
        "W": camera_params["W"]
    }

    total_frames = len(camera_params["R"])
    keyframe_indices = [k for k in range(total_frames) if k % 24 == 0]

    # 1. 先处理所有的关键帧
    for idx in keyframe_indices:
        if idx < total_frames:
            real_camera_params["R"].append(camera_params["R"][idx])
            real_camera_params["T"].append(camera_params["T"][idx])

    # 2. 处理中间帧
    for i in range(total_frames):
        if i not in keyframe_indices:
            mid_camera_params["R"].append(camera_params["R"][i])
            mid_camera_params["T"].append(camera_params["T"][i])

    # 3. 如果有图像数据，只分配给关键帧
    if "images" in camera_params:
        # 确保我们只使用与关键帧一样多的图像
        available_images = len(camera_params["images"])
        if len(keyframe_indices) == available_images:
            print('load correct number of images')
            frames_to_use = len(keyframe_indices)
        else:
            print('load incorrect number of images')
            frames_to_use = min(len(keyframe_indices), available_images)
            real_camera_params['images'] = camera_params['images'][:frames_to_use]

        assert len(real_camera_params["R"]) == len(real_camera_params['images'])
    return real_camera_params, mid_camera_params

def read_camera_params_without_mid(file_path):
    """
    从不包含中间帧的文本文件中读取相机参数

    Args:
        file_path: 相机参数文本文件路径

    Returns:
        camera_params: 包含相机参数的字典，格式为:
            {
                "R": [r1, r2, ..., rn],         # 旋转矩阵列表
                "T": [t1, t2, ..., tn],         # 平移向量列表
                "focals": [fx, fy],             # 焦距
                "H": height,                    # 图像高度
                "W": width,                     # 图像宽度
                "images": [                     # 图像信息
                    {"name": "img1.jpg", "path": "/path/to/img1.jpg"},
                    ...
                ]
            }
    """
    camera_params = {
        "R": [],
        "T": [],
        "focals": None,
        "H": None,
        "W": None,
        "images": []
    }

    # 临时存储图像名称和路径
    image_names = []
    image_paths = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # 处理每一行
            line_idx = 0
            while line_idx < len(lines):
                line = lines[line_idx].strip()

                if line.startswith("R:"):
                    try:
                        camera_params["R"].append(json.loads(line[2:]))
                    except json.JSONDecodeError as e:
                        print(f"无法解析旋转矩阵: {line}. 错误: {e}")
                        # 尝试修复格式并重试
                        fixed_line = line[2:].replace("'", "\"")
                        try:
                            camera_params["R"].append(json.loads(fixed_line))
                            print("已修复并解析")
                        except:
                            print("修复失败，跳过此行")

                elif line.startswith("T:"):
                    try:
                        camera_params["T"].append(json.loads(line[2:]))
                    except json.JSONDecodeError as e:
                        print(f"无法解析平移向量: {line}. 错误: {e}")
                        # 尝试修复格式并重试
                        fixed_line = line[2:].replace("'", "\"")
                        try:
                            camera_params["T"].append(json.loads(fixed_line))
                            print("已修复并解析")
                        except:
                            print("修复失败，跳过此行")

                elif line.startswith("focals:"):
                    try:
                        camera_params["focals"] = json.loads(line[7:])
                    except json.JSONDecodeError:
                        # 尝试修复格式并重试
                        fixed_line = line[7:].replace("'", "\"")
                        try:
                            camera_params["focals"] = json.loads(fixed_line)
                        except:
                            # 尝试直接解析为列表
                            try:
                                nums = [float(x) for x in line[7:].strip('[]').split(',')]
                                if len(nums) == 2:
                                    camera_params["focals"] = nums
                                else:
                                    print(f"焦距格式错误: {line}")
                            except:
                                print(f"无法解析焦距: {line}")

                elif line.startswith("H:"):
                    try:
                        camera_params["H"] = int(float(line[2:].strip()))
                    except ValueError:
                        print(f"无法解析图像高度: {line}")

                elif line.startswith("W:"):
                    try:
                        camera_params["W"] = int(float(line[2:].strip()))
                    except ValueError:
                        print(f"无法解析图像宽度: {line}")

                elif line.startswith("image_name:"):
                    image_name = line[11:].strip()
                    image_names.append(image_name)

                elif line.startswith("image_path:"):
                    image_path = line[11:].strip()
                    image_paths.append(image_path)

                line_idx += 1

        # 组织图像信息
        for i in range(min(len(image_names), len(image_paths))):
            camera_params["images"].append({
                "name": image_names[i],
                "path": image_paths[i]
            })

        # 检查R、T和images的数量是否一致
        num_r = len(camera_params["R"])
        num_t = len(camera_params["T"])
        num_imgs = len(camera_params["images"])

        if num_r != num_t or num_r != num_imgs:
            print(f"警告: 参数数量不匹配 - R: {num_r}, T: {num_t}, images: {num_imgs}")

        # 确认数据完整性
        if not camera_params["R"] or not camera_params["T"] or not camera_params["images"]:
            print("警告: 缺少必要的相机参数")

        if camera_params["focals"] is None:
            print("警告: 缺少焦距信息")

        if camera_params["H"] is None or camera_params["W"] is None:
            print("警告: 缺少图像尺寸信息")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取相机参数时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    return camera_params

def verify_camera_params(camera_params):
    """
    验证相机参数的完整性和一致性

    Args:
        camera_params: 相机参数字典

    Returns:
        bool: 参数是否有效
    """
    # 基本检查
    if camera_params is None:
        print("相机参数为空")
        return False

    # 检查必要的键
    required_keys = ["R", "T", "focals", "H", "W", "images"]
    for key in required_keys:
        if key not in camera_params or camera_params[key] is None:
            print(f"缺少必要的相机参数: {key}")
            return False

    # 检查数量一致性
    num_r = len(camera_params["R"])
    num_t = len(camera_params["T"])
    num_imgs = len(camera_params["images"])

    if num_r != num_t:
        print(f"旋转矩阵(R)和平移向量(T)的数量不一致: R={num_r}, T={num_t}")
        return False

    if num_r != num_imgs:
        print(f"相机姿态和图像的数量不一致: 姿态={num_r}, 图像={num_imgs}")
        return False

    # 检查数据类型
    for i, r in enumerate(camera_params["R"]):
        if not isinstance(r, list) or len(r) != 3 or not all(isinstance(row, list) and len(row) == 3 for row in r):
            print(f"旋转矩阵 #{i} 格式无效: {r}")
            return False

    for i, t in enumerate(camera_params["T"]):
        if not isinstance(t, list) or len(t) != 3:
            print(f"平移向量 #{i} 格式无效: {t}")
            return False

    # 检查焦距
    if not isinstance(camera_params["focals"], list) or len(camera_params["focals"]) != 2:
        print(f"焦距格式无效: {camera_params['focals']}")
        return False

    # 检查图像尺寸
    if not isinstance(camera_params["H"], int) or not isinstance(camera_params["W"], int):
        print(f"图像尺寸无效: H={camera_params['H']}, W={camera_params['W']}")
        return False

    print(f"相机参数验证通过: {num_r}个姿态, {num_imgs}个图像")
    return True