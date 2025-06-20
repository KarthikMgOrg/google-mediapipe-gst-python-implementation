import cv2
import mediapipe as mp
from multiprocessing import shared_memory, Process, Queue
import numpy as np

FRAME_SHAPE = (720, 1280, 3)
FRAME_DTYPE = np.uint8
BATCH_SIZE = 5
FRAME_RATE = 30  # frames per second, adjust based on actual video source


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

    while True:
        signal, valid_frames, batch_number = control_queue_facemesh.get()
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
                    print(f"Skipping frame {idx} due to zero eye height.")
                    continue

                vertical_position = iris_vertical_offset / eye_height

                absolute_frame_idx = batch_number * BATCH_SIZE + idx
                video_timestamp = absolute_frame_idx / FRAME_RATE

                if relative_position < 0.2:
                    gaze_direction = "Looking Away Left"
                elif relative_position > 0.8:
                    gaze_direction = "Looking Away Right"
                elif vertical_position < 0.3:
                    gaze_direction = "Looking Up"
                elif vertical_position > 0.8:
                    gaze_direction = "Looking Down"
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

        # Save back
        result_queue["eye_out_of_view"] = dict(existing_counter)
        # print('end of batch')
    face_mesh.close()
    shm.close()
    print('Finished consumer')
            

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

    while True:
        signal, valid_frames, batch_number = control_queue_mood.get()
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

                # Basic mood analysis using mouth landmarks
                mouth_left = face_landmarks.landmark[61]
                mouth_right = face_landmarks.landmark[291]
                mouth_top = face_landmarks.landmark[13]
                mouth_bottom = face_landmarks.landmark[14]

                mouth_width = abs(int(mouth_right.x * iw) - int(mouth_left.x * iw))
                mouth_height = abs(int(mouth_bottom.y * ih) - int(mouth_top.y * ih))

                smile_ratio = mouth_width / (mouth_height + 1e-6)  # Avoid division by zero

                absolute_frame_idx = batch_number * BATCH_SIZE + idx
                video_timestamp = absolute_frame_idx / FRAME_RATE

                if smile_ratio > 4.0:
                    mood = "Happy"
                else:
                    mood = "Neutral"

                results_for_batch.append({
                    'frame': absolute_frame_idx,
                    'mood': mood,
                    'video_timestamp': video_timestamp
                })
            else:
                continue

        if "mood_analysis" not in result_queue:
            result_queue["mood_analysis"] = []

        current_results = result_queue["mood_analysis"]
        current_results.extend(results_for_batch)
        result_queue["mood_analysis"] = current_results

    face_mesh.close()
    shm.close()
    print('Finished mood analysis consumer')
