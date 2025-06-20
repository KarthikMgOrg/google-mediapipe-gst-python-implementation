# stream_video.py

from gi.repository import Gst
import gi
import threading
import numpy as np
from queue import Queue
import os
import cv2
import matplotlib.pyplot as plt

os.environ['GST_GL_API'] = 'disabled'

gi.require_version('Gst', '1.0')

Gst.init(None)


def producer(pipeline, appsink, frame_queue):
    print('started producer')
    pipeline.set_state(Gst.State.PLAYING)

    while True:
        sample = appsink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value('width')
            height = caps.get_structure(0).get_value('height')

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                try:
                    frame = np.frombuffer(map_info.data, np.uint8)
                    expected_size = height * width * 3
                    if frame.size == expected_size:
                        frame = frame.reshape((height, width, 3))
                        if not frame_queue.full():
                            frame_queue.put(frame)
                    else:
                        print(
                            f"Warning: Frame size mismatch, expected {expected_size} bytes, got {frame.size}")
                finally:
                    buffer.unmap(map_info)
            else:
                break
        else:
            break

    frame_queue.put(None)
    pipeline.set_state(Gst.State.NULL)


def consumer_with_matplotlib(frame_queue):
    print('Started consumer on main thread')

    plt.ion()
    fig, ax = plt.subplots()
    img_display = None

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        try:
            print(frame.shape)
            resized_frame = cv2.resize(frame, (960, 540))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            if img_display is None:
                img_display = ax.imshow(rgb_frame)
            else:
                img_display.set_data(rgb_frame)

            plt.pause(0.001)

        except Exception:
            import traceback
            traceback.print_exc()
            continue

    plt.ioff()
    plt.show()
    print('Finished consumer')


def setup_pipeline(url):
    gst_pipeline_str = (
        f'souphttpsrc location={url} ! matroskademux ! vp9dec ! videoconvert ! '
        f'video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=sink max-buffers=1 drop=true'
    )

    pipeline = Gst.parse_launch(gst_pipeline_str)
    appsink = pipeline.get_by_name('sink')
    appsink.set_property('emit-signals', True)
    appsink.set_property('sync', False)

    return pipeline, appsink


if __name__ == "__main__":
    print('Started Streaming')

    url = 'https://d1e6cahdfvkipq.cloudfront.net/videos/hyring-1749562600257'
    queue_size = 10
    frame_queue = Queue(maxsize=queue_size)

    pipeline, appsink = setup_pipeline(url)

    producer_thread = threading.Thread(
        target=producer, args=(pipeline, appsink, frame_queue))
    producer_thread.start()

    # Directly run consumer in the main thread
    consumer_with_matplotlib(frame_queue)

    producer_thread.join()
    print('Finished Streaming')
