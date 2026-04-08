"""Read image frames from ROS 1 .bag files (e.g. Intel RealSense recordings)."""

from __future__ import annotations

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


def count_bag_frames(bag_path: str) -> int:
    """Count image frames in a .bag file."""
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        conn = _preferred_image_connection(reader.connections)
        if conn is None:
            return 0
        return sum(1 for _ in reader.messages(connections=[conn]))


def extract_frame(bag_path: str, frame_index: int) -> np.ndarray | None:
    """Extract the frame at *frame_index* from the preferred image topic."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        conn = _preferred_image_connection(reader.connections)
        if conn is None:
            return None
        for i, (connection, _ts, rawdata) in enumerate(
            reader.messages(connections=[conn])
        ):
            if i == frame_index:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                if "CompressedImage" in connection.msgtype:
                    return cv2.imdecode(
                        np.frombuffer(bytes(msg.data), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                return _raw_image_to_bgr(msg)
    return None


def _raw_image_to_bgr(msg) -> np.ndarray:
    """Convert a ``sensor_msgs/Image`` message to a BGR numpy array."""
    h, w = msg.height, msg.width
    data = bytes(msg.data)
    enc = msg.encoding

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
    if enc in ("16UC1", "32FC1"):
        dtype = np.uint16 if enc == "16UC1" else np.float32
        raw = np.frombuffer(data, dtype=dtype).reshape(h, w)
        norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    channels = max(len(data) // (h * w), 1)
    arr = np.frombuffer(data, np.uint8).reshape(h, w, channels)
    if channels == 1:
        return cv2.cvtColor(arr.squeeze(), cv2.COLOR_GRAY2BGR)
    return arr[:, :, :3].copy()
