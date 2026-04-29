import cv2
import time
import numpy as np
import mediapipe as mp
from collections import namedtuple

model_path = 'pose_landmarker_lite.task'

LandmarkData = namedtuple('LANDMARK', ['name', 'x', 'y', 'z', 'vis'])
LANDMARKS = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):  # pyright: ignore[reportInvalidTypeForm]
    pass


latest_result = None


def store_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):  # pyright: ignore[reportInvalidTypeForm]
    global latest_result
    latest_result = result


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=store_result,
)


def get_all_landmarks(result):
    if ((not result) or (not result.pose_landmarks)):
        return None

    pose = result.pose_landmarks[0]

    return {
        idx: LandmarkData(LANDMARKS[idx], lndmrk.x, lndmrk.y, lndmrk.z, lndmrk.visibility)
        for idx, lndmrk in enumerate(pose)
    }


def get_landmark(result, name):
    lndmrks = get_all_landmarks(result)
    if (not lndmrks):
        return None

    for item in lndmrks.values():
        if item.name == name:
            return item

    return None


def get_angle(first, mid, end):
    if ((not first) or (not mid) or (not end)):
        return 0

    radians = np.arctan2(end.y - mid.y, end.x - mid.x) - np.arctan2(first.y - mid.y, first.x - mid.x)
    angle = np.abs(radians * 180.0 / np.pi)

    if (angle > 180.0):
        angle = 360 - angle

    return angle


def draw_landmarks_on_frame(frame, result):
    if ((not result) or (not result.pose_landmarks)):
        return frame

    height, width, _ = frame.shape

    for pose in result.pose_landmarks:
        connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

        for connection in connections:
            start = pose[connection.start]
            end = pose[connection.end]

            x1, y1 = int(start.x * width), int(start.y * height)
            x2, y2 = int(end.x * width), int(end.y * height)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for landmark in pose:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return frame


with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if (not ret):
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
        frame = draw_landmarks_on_frame(frame, latest_result)

        win_rect = cv2.getWindowImageRect('Mediapipe Feed')
        win_width, win_height = win_rect[2], win_rect[3]

        if (win_width > 0 and win_height > 0):
            height, width = frame.shape[:2]
            scale = min(win_width / width, win_height / height)

            new_width = int(width * scale)
            new_height = int(height * scale)

            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((win_height, win_width, 3), dtype=np.uint8)

            x_offset = (win_width - new_width) // 2
            y_offset = (win_height - new_height) // 2
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

            frame = canvas

        if (latest_result):
            first = get_landmark(latest_result, "right_shoulder")
            mid = get_landmark(latest_result, "right_elbow")
            end = get_landmark(latest_result, "right_wrist")

            threshold = 0.6
            if (
                first and mid and end
                and (first.vis or 0) > threshold
                and (mid.vis or 0) > threshold
                and (end.vis or 0) > threshold
            ):
                angle = get_angle(first, mid, end)

                print(get_angle(first, mid, end))
            else:
                print("Landmarks not visible...")

        cv2.imshow('Mediapipe Feed', frame)

        if (cv2.waitKey(10) & 0xFF == ord('q')) or (cv2.getWindowProperty('Mediapipe Feed', cv2.WND_PROP_VISIBLE) < 1):
            break

    cap.release()
    cv2.destroyAllWindows()
