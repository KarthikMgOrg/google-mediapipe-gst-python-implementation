import logging
import sys
import json
import numpy as np
from multiprocessing import shared_memory, Queue, Process, Manager
from collections import defaultdict
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from mediapipe_api.utils import gst_frame_consumer, gst_frame_producer

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python process_worker.py <video_url>"}), file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    manager = Manager()

    benchmark_result = manager.dict({
        'start_time': None,
        'end_time': None,
        'frame_count': 0
    })
    result_queue = manager.dict({
        "eye_out_of_view": [],
        "overall_mood": ""
    })

    frame_shape = (720, 1280, 3)
    batch_size = 5
    total_size = np.prod((batch_size,) + frame_shape) * np.dtype(np.uint8).itemsize

    shm = shared_memory.SharedMemory(create=True, size=total_size)
    control_queues = [Queue(), Queue()]

    try:
        producer_process = Process(target=gst_frame_producer, args=(url, benchmark_result, shm.name, frame_shape, control_queues, batch_size))
        consumer_process = Process(target=gst_frame_consumer, args=(shm.name, control_queues, result_queue, 1))

        producer_process.start()
        consumer_process.start()

        producer_process.join(timeout=120)
        if producer_process.is_alive():
            producer_process.terminate()

        consumer_process.join(timeout=120)
        if consumer_process.is_alive():
            consumer_process.terminate()

        shm.close()
        shm.unlink()

        # Post-processing
        moods = result_queue.get("mood_analysis", [])
        sentiment_by_timeblock = defaultdict(list)
        for entry in moods:
            block = int(entry['video_timestamp'] // 5)
            sentiment_by_timeblock[block].append(entry['mood_score'])

        sentiment_summary = {
            f"{block * 5}-{(block + 1) * 5}s": round(sum(scores) / len(scores), 2)
            for block, scores in sentiment_by_timeblock.items()
        }

        result_queue['sentiment_over_time'] = sentiment_summary
        result_queue.pop('mood_analysis', None)

        start_time = benchmark_result.get('start_time')
        end_time = benchmark_result.get('end_time')
        total_time = (end_time - start_time) if start_time and end_time else 0
        fps = benchmark_result['frame_count'] / total_time if total_time else 0

        output = {
            "frames_processed": benchmark_result['frame_count'],
            "total_time_seconds": total_time,
            "processing_fps": fps,
            "results": dict(result_queue),
        }

        try:
            json.dumps(output)  # validate JSON
        except Exception as e:
            print("Invalid JSON structure:", str(e), file=sys.stderr)
            print(repr(output), file=sys.stderr)
            sys.exit(1)
        print(json.dumps(output), flush=True)
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()