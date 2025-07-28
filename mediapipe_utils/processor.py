import cv2
import mediapipe as mp
from multiprocessing import shared_memory, Process, Queue
import numpy as np
import os
import json

FRAME_SHAPE = (720, 1280, 3)
FRAME_DTYPE = np.uint8
BATCH_SIZE = 5


def get_video_fps_from_file(video_path="https://d1e6cahdfvkipq.cloudfront.net/videos/hyring-1749562600257"):
    import tempfile
    import urllib.request

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            urllib.request.urlretrieve(video_path, tmp_file.name)
            cap = cv2.VideoCapture(tmp_file.name)
            if not cap.isOpened():
                print(f"Error: Could not open downloaded video file")
                return 30
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps else 30
    except Exception as e:
        print(f"Error downloading or reading video: {e}")
        return 30


FRAME_RATE = get_video_fps_from_file()


def process_batches_with_facemesh(control_queue_facemesh, shm_name, result_queue):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Important for iris tracking
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    shm = shared_memory.SharedMemory(name=shm_name)
    batch_buffer = np.ndarray(
        (BATCH_SIZE,) + FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)

    import queue
    while True:
        try:
            signal, valid_frames, batch_number = control_queue_facemesh.get(
                timeout=60)
        except queue.Empty:
            # print("[Facemesh] Queue timed out. Exiting.")
            break
        if signal == 'stop':
            break  # Clean exit when producer signals completion

        results_for_batch = []
        for idx in range(valid_frames):
            frame = batch_buffer[idx].copy()
            # print(
            #     f"processed frame {idx} with shape: {frame.shape}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ih, iw, _ = frame.shape

                left_iris_center = face_landmarks.landmark[468]
                left_iris_x = int(left_iris_center.x * iw)
                left_iris_y = int(left_iris_center.y * ih)

                left_eye_inner = face_landmarks.landmark[133]
                left_eye_outer = face_landmarks.landmark[33]

                left_eye_inner_x = int(left_eye_inner.x * iw)
                left_eye_outer_x = int(left_eye_outer.x * iw)

                eye_width = abs(left_eye_inner_x - left_eye_outer_x)
                iris_offset = left_iris_x - left_eye_outer_x

                relative_position = iris_offset / eye_width

                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]

                left_eye_top_y = int(left_eye_top.y * ih)
                left_eye_bottom_y = int(left_eye_bottom.y * ih)

                eye_height = abs(left_eye_top_y - left_eye_bottom_y)
                iris_vertical_offset = left_iris_y - left_eye_top_y

                if eye_height == 0:
                    # print(f"Skipping frame {idx} due to zero eye height.")
                    continue

                vertical_position = iris_vertical_offset / eye_height

                absolute_frame_idx = batch_number * BATCH_SIZE + idx
                video_timestamp = round(absolute_frame_idx / FRAME_RATE, 3)

                if relative_position < 0.3:
                    gaze_direction = "Looking Away Left"
                    # print(f"Detected gaze LEFT at frame {absolute_frame_idx}, relative_position={relative_position:.2f}")
                elif relative_position > 0.7:
                    gaze_direction = "Looking Away Right"
                    # print(f"Detected gaze RIGHT at frame {absolute_frame_idx}, relative_position={relative_position:.2f}")
                # elif vertical_position < 0.1:
                #     gaze_direction = "Looking Up"
                # elif vertical_position > 0.9:
                #     gaze_direction = "Looking Down"
                else:
                    continue

                results_for_batch.append({
                    'frame': absolute_frame_idx,
                    'gaze_direction': gaze_direction,
                    'video_timestamp': video_timestamp  # seconds into the video
                })
            else:
                continue

        # result_queue.put({'results': results_for_batch})
        from collections import Counter
        if "eye_out_of_view" not in result_queue:
            result_queue["eye_out_of_view"] = {}
        # Load existing counter
        existing_counter = Counter(result_queue["eye_out_of_view"])
        # Update with current batch
        face_out_counter = Counter([item['video_timestamp']
                                    for item in results_for_batch])
        existing_counter.update(face_out_counter)
        # print(existing_counter, " existing counter")

        # Save back
        result_queue["eye_out_of_view"] = dict(existing_counter)
        # print('end of batch')

    from .utils import group_gazes
    updated_eye_out_of_view = group_gazes(result_queue["eye_out_of_view"])
    result_queue['eye_out_of_view'] = updated_eye_out_of_view
    face_mesh.close()
    shm.close()
    # print('Finished consumer')


def analyze_sentiment_score_from_face(landmarks, iw, ih):
    """
    Returns sentiment score between 0 and 100.
    100 = very positive (happy), 0 = very negative (angry)
    """
    mouth_left = landmarks.landmark[61]
    mouth_right = landmarks.landmark[291]
    mouth_top = landmarks.landmark[13]
    mouth_bottom = landmarks.landmark[14]

    mouth_width = abs(int(mouth_right.x * iw) - int(mouth_left.x * iw))
    mouth_height = abs(int(mouth_bottom.y * ih) - int(mouth_top.y * ih))

    left_eye_top = landmarks.landmark[159]
    left_eye_bottom = landmarks.landmark[145]
    right_eye_top = landmarks.landmark[386]
    right_eye_bottom = landmarks.landmark[374]

    left_eye_height = abs(int(left_eye_bottom.y * ih) -
                          int(left_eye_top.y * ih))
    right_eye_height = abs(
        int(right_eye_bottom.y * ih) - int(right_eye_top.y * ih))
    avg_eye_height = (left_eye_height + right_eye_height) / 2.0

    left_eyebrow = landmarks.landmark[70]
    right_eyebrow = landmarks.landmark[300]
    brow_y = (left_eyebrow.y + right_eyebrow.y) / 2.0 * ih
    eye_center_y = (left_eye_top.y + right_eye_top.y) / 2.0 * ih
    brow_eye_gap = eye_center_y - brow_y

    smile_ratio = mouth_width / (mouth_height + 1e-6)

    if smile_ratio > 3.5:
        return 100  # Happy
    elif brow_eye_gap < 10 and smile_ratio < 2.0:
        return 10  # Angry
    elif avg_eye_height > 20 and mouth_height > 15:
        return 75  # Surprised
    elif mouth_height > 10 and smile_ratio < 2.0:
        return 25  # Sad
    else:
        return 50  # Neutral


def process_batches_with_mood_analysis(control_queue_mood, shm_name, result_queue):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    shm = shared_memory.SharedMemory(name=shm_name)
    batch_buffer = np.ndarray(
        (BATCH_SIZE,) + FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)
    mood_scores_accumulator = []

    import queue
    while True:
        try:
            signal, valid_frames, batch_number = control_queue_mood.get(
                timeout=60)
        except queue.Empty:
            # print("[Mood] Queue timed out. Exiting.")
            break
        if signal == 'stop':
            break  # Clean exit when producer signals completion

        results_for_batch = []
        for idx in range(valid_frames):
            frame = batch_buffer[idx].copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ih, iw, _ = frame.shape

                absolute_frame_idx = batch_number * BATCH_SIZE + idx
                video_timestamp = absolute_frame_idx / FRAME_RATE

                mood_score = analyze_sentiment_score_from_face(
                    face_landmarks, iw, ih)
                mood_scores_accumulator.append(mood_score)

                results_for_batch.append({
                    'frame': absolute_frame_idx,
                    'mood_score': mood_score,
                    'video_timestamp': video_timestamp
                })
            else:
                continue

        if "mood_analysis" not in result_queue:
            result_queue["mood_analysis"] = []

        current_results = result_queue["mood_analysis"]
        current_results.extend(results_for_batch)
        result_queue["mood_analysis"] = current_results

    # Calculate and store overall mood score before closing
    if mood_scores_accumulator:
        avg_mood = sum(mood_scores_accumulator) / len(mood_scores_accumulator)
        result_queue["overall_mood_score"] = round(avg_mood, 2)
        # Calculate percentage mood map
        mood_counts = {"happy": 0, "neutral": 0, "sad": 0, "angry": 0}
        for score in mood_scores_accumulator:
            if score == 100:
                mood_counts["happy"] += 1
            elif score == 50:
                mood_counts["neutral"] += 1
            elif score == 25:
                mood_counts["sad"] += 1
            elif score == 10:
                mood_counts["angry"] += 1
        total = sum(mood_counts.values())
        mood_map = {k: round((v / total) * 100, 2)
                    if total else 0 for k, v in mood_counts.items()}
        result_queue["mood_map"] = mood_map
    else:
        result_queue["overall_mood_score"] = None

    face_mesh.close()
    shm.close()
    # print('Finished mood analysis consumer')


def process_batches_with_head_pose(control_queue, shm_name, result_queue):
    import mediapipe as mp
    import numpy as np
    import cv2
    from multiprocessing import shared_memory

    mp_face_mesh = mp.solutions.face_mesh

    shm = shared_memory.SharedMemory(name=shm_name)
    frame_shape = (720, 1280, 3)
    batch_size = 5
    batch_buffer = np.ndarray(
        (batch_size,) + frame_shape, dtype=np.uint8, buffer=shm.buf)

    import queue
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            try:
                command, batch_index, batch_count = control_queue.get(
                    timeout=60)
            except queue.Empty:
                print("[HeadPose] Queue timed out. Exiting.")
                break
            if command == 'stop':
                break
            for i in range(batch_index):
                frame = batch_buffer[i]
                results = face_mesh.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract nose, eyes and ears for head pose estimation
                        # Example landmark
                        nose_tip = face_landmarks.landmark[1]
                        nose_tip_dict = nose_tip.x, nose_tip.y, nose_tip.z
                        result_queue['headpose_results'] = {
                            'process': 'head_pose', 'nose_tip': nose_tip_dict}
                        # result_queue.put({'process': 'head_pose', 'batch': batch_count, 'frame': i, 'nose_tip': (
                        #     nose_tip.x, nose_tip.y, nose_tip.z)})

    shm.close()
