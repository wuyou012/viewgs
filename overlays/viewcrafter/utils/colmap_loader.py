import collections
import struct

import numpy as np


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
    ])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            assert model == "PINHOLE", (
                "While the loader supports other types, the rest of the code assumes PINHOLE")
            width, height = int(elems[2]), int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_extrinsics_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            elems = fid.readline().split()
            xys = np.column_stack([
                tuple(map(float, elems[0::3])),
                tuple(map(float, elems[1::3])),
            ])
            point3D_ids = np.array(tuple(map(int, elems[2::3])))
            images[image_id] = Image(
                image_id, qvec, tvec, camera_id, image_name, xys, point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id, width, height = camera_properties
            model = CAMERA_MODEL_IDS[model_id]
            params = read_next_bytes(fid, 8 * model.num_params, "d" * model.num_params)
            cameras[camera_id] = Camera(
                camera_id, model.model_name, width, height, np.array(params))
    return cameras


def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = properties[0]
            qvec = np.array(properties[1:5])
            tvec = np.array(properties[5:8])
            camera_id = properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            values = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([
                tuple(map(float, values[0::3])),
                tuple(map(float, values[1::3])),
            ])
            point3D_ids = np.array(tuple(map(int, values[2::3])))
            images[image_id] = Image(
                image_id, qvec, tvec, camera_id, image_name, xys, point3D_ids)
    return images
