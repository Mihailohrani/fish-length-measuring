"""Read image frames from ROS 1 .bag files (e.g. Intel RealSense recordings)."""

from __future__ import annotations

import struct

import cv2
import numpy as np


def _preferred_image_connection(connections):
    """Pick the best image connection, preferring color over depth/infrared."""
    image_conns = [c for c in connections if "Image" in c.msgtype]
    if not image_conns:
        return None
    for keyword in ("color", "rgb", "bgr"):
        for c in image_conns:
            if keyword in c.topic.lower():
                return c
    skip = ("depth", "infrared", "ir_", "confidence")
    for c in image_conns:
        if not any(k in c.topic.lower() for k in skip):
            return c
    return image_conns[0]


def get_depth_intrinsics(bag_path: str) -> dict | None:
    """Extract camera intrinsics and depth units from a RealSense .bag file.

    Returns a dict with keys ``fx``, ``fy``, ``cx``, ``cy``, ``depth_units``
    (meters per raw depth tick), or ``None`` if the data is unavailable.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        cam_conns = [c for c in reader.connections if "camera_info" in c.topic.lower()]
        du_conns = [c for c in reader.connections if "Depth_Units/value" in c.topic]

        if not cam_conns:
            return None

        intrinsics: dict = {}

        for _conn, _ts, rawdata in reader.messages(connections=cam_conns[:1]):
            msg = typestore.deserialize_ros1(rawdata, _conn.msgtype)
            k = msg.K
            intrinsics = {"fx": float(k[0]), "fy": float(k[4]),
                          "cx": float(k[2]), "cy": float(k[5]),
                          "depth_units": 0.001}
            break

        if du_conns:
            for _conn, _ts, rawdata in reader.messages(connections=du_conns[:1]):
                msg = typestore.deserialize_ros1(rawdata, _conn.msgtype)
                intrinsics["depth_units"] = float(msg.data)
                break

        return intrinsics if intrinsics else None


def extract_raw_depth(bag_path: str, frame_index: int) -> np.ndarray | None:
    """Extract the raw uint16 depth array for *frame_index* (no colormap)."""
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        conn = _preferred_image_connection(reader.connections)
        if conn is None:
            return None
        for i, (_connection, _ts, rawdata) in enumerate(
            reader.messages(connections=[conn])
        ):
            if i == frame_index:
                msg = _parse_ros1_image(rawdata)
                enc = msg["encoding"]
                h, w = msg["height"], msg["width"]
                data = msg["data"]
                if enc in ("mono16", "16UC1"):
                    return np.frombuffer(data, dtype=np.uint16).reshape(h, w)
                if enc == "32FC1":
                    return np.frombuffer(data, dtype=np.float32).reshape(h, w)
                return None
    return None


def has_color_stream(bag_path: str) -> bool:
    """Return True if the bag contains a color (non-depth/IR) image topic."""
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        image_conns = [c for c in reader.connections if "Image" in c.msgtype]
        for c in image_conns:
            topic = c.topic.lower()
            if any(k in topic for k in ("color", "rgb", "bgr")):
                return True
        skip = ("depth", "infrared", "ir_", "confidence")
        for c in image_conns:
            if not any(k in c.topic.lower() for k in skip):
                return True
        return False


def count_bag_frames(bag_path: str) -> int:
    """Count image frames in a .bag file."""
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        conn = _preferred_image_connection(reader.connections)
        if conn is None:
            return 0
        return sum(1 for _ in reader.messages(connections=[conn]))


def _parse_ros1_image(rawdata: bytes) -> dict:
    """Parse a sensor_msgs/Image from raw ROS 1 bytes.

    Works with standard ROS bags and RealSense recordings that cause
    rosbags' ``deserialize_ros1`` to fail on trailing bytes.
    """
    data = bytes(rawdata)
    pos = 0

    # std_msgs/Header
    pos += 4  # seq
    pos += 8  # stamp (sec + nsec)
    frame_id_len = struct.unpack_from("<I", data, pos)[0]
    pos += 4 + frame_id_len

    height = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    width = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    encoding_len = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    encoding = data[pos : pos + encoding_len].decode()
    pos += encoding_len

    pos += 1  # is_bigendian
    pos += 4  # step

    data_len = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    image_data = data[pos : pos + data_len]

    return {"height": height, "width": width, "encoding": encoding, "data": image_data}


def extract_frame(bag_path: str, frame_index: int) -> np.ndarray | None:
    """Extract the frame at *frame_index* from the preferred image topic."""
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        conn = _preferred_image_connection(reader.connections)
        if conn is None:
            return None
        for i, (connection, _ts, rawdata) in enumerate(
            reader.messages(connections=[conn])
        ):
            if i == frame_index:
                if "CompressedImage" in connection.msgtype:
                    return cv2.imdecode(
                        np.frombuffer(bytes(rawdata), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                msg = _parse_ros1_image(rawdata)
                return _image_to_bgr(msg)
    return None


def _image_to_bgr(msg: dict) -> np.ndarray:
    """Convert a parsed sensor_msgs/Image dict to a BGR numpy array."""
    h, w = msg["height"], msg["width"]
    data = msg["data"]
    enc = msg["encoding"]

    if enc == "rgb8":
        return cv2.cvtColor(
            np.frombuffer(data, np.uint8).reshape(h, w, 3), cv2.COLOR_RGB2BGR
        )
    if enc == "bgr8":
        return np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()
    if enc == "bgra8":
        return cv2.cvtColor(
            np.frombuffer(data, np.uint8).reshape(h, w, 4), cv2.COLOR_BGRA2BGR
        )
    if enc == "rgba8":
        return cv2.cvtColor(
            np.frombuffer(data, np.uint8).reshape(h, w, 4), cv2.COLOR_RGBA2BGR
        )
    if enc in ("mono8", "8UC1"):
        return cv2.cvtColor(
            np.frombuffer(data, np.uint8).reshape(h, w), cv2.COLOR_GRAY2BGR
        )
    if enc in ("mono16", "16UC1", "32FC1"):
        dtype = np.float32 if enc == "32FC1" else np.uint16
        raw = np.frombuffer(data, dtype=dtype).reshape(h, w)
        norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    channels = max(len(data) // (h * w), 1)
    arr = np.frombuffer(data, np.uint8).reshape(h, w, channels)
    if channels == 1:
        return cv2.cvtColor(arr.squeeze(), cv2.COLOR_GRAY2BGR)
    return arr[:, :, :3].copy()
