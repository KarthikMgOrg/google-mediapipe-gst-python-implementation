import mediapipe as mp
import gi
import threading
import numpy as np
from queue import Queue
import os
import cv2
import matplotlib.pyplot as plt
from gi.repository import Gst
import numpy as np
from multiprocessing import shared_memory, Process, Queue
import time
from mediapipe_utils.processor import process_batches_with_facemesh, process_batches_with_mood_analysis, process_batches_with_head_pose

os.environ['GST_GL_API'] = 'disabled'
gi.require_version('Gst', '1.0')
Gst.init(None)

# Configuration
FRAME_SHAPE = (720, 1280, 3)
FRAME_DTYPE = np.uint8
BATCH_SIZE = 5


def gst_frame_producer(url, benchmark_result, shm_name, frame_shape, control_queues, batch_size=5):
    # print('started producer')
    benchmark_result['start_time'] = time.time()

    # Create pipeline and elements
    pipeline = Gst.Pipeline.new("pipeline")
    src = Gst.ElementFactory.make("souphttpsrc", "source")
    src.set_property("location", url)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    demuxer = Gst.ElementFactory.make("matroskademux", "demuxer")

    queue2 = Gst.ElementFactory.make("queue", "queue2")
    decoder = Gst.ElementFactory.make("vp9dec", "decoder")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    convert = Gst.ElementFactory.make("videoconvert", "convert")
    scale = Gst.ElementFactory.make("videoscale", "scale")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps = Gst.Caps.from_string("video/x-raw, width=1280, height=720, format=BGR")
    capsfilter.set_property("caps", caps)

    sink = Gst.ElementFactory.make("appsink", "sink")
    sink.set_property("emit-signals", True)
    sink.set_property("sync", False)
    sink.set_property('max-buffers', 30)
    sink.set_property('drop', True)

    for elem in [src, queue1, demuxer, queue2, decoder, queue3, convert, scale, capsfilter, sink]:
        if not elem:
            # print(f"Element creation failed.")
            return
        pipeline.add(elem)

    src.link(queue1)
    queue1.link(demuxer)

    queue2.link(decoder)
    decoder.link(queue3)
    queue3.link(convert)
    convert.link(scale)
    scale.link(capsfilter)
    capsfilter.link(sink)

    def on_pad_added(demuxer, pad):
        # print("Dynamic pad created, linking demuxer to decoder...")
        sink_pad = queue2.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    demuxer.connect("pad-added", on_pad_added)

    shm = shared_memory.SharedMemory(name=shm_name)
    batch_buffer = np.ndarray((batch_size,) + frame_shape, dtype=np.uint8, buffer=shm.buf)

    batch_index = 0
    frame_count = 0
    pipeline.set_state(Gst.State.PLAYING)

    # print('Pipeline set to PLAYING. Waiting for bus messages...')
    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(
            Gst.SECOND * 5, Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.STATE_CHANGED)

        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                # print(f"Error received from element {msg.src.get_name()}: {err}")
                # print(f"Debugging information: {debug}")
                return
            elif msg.type == Gst.MessageType.EOS:
                # print("End-Of-Stream reached.")
                return
            elif msg.type == Gst.MessageType.STATE_CHANGED:
                old_state, new_state, pending_state = msg.parse_state_changed()
                if msg.src == pipeline:
                    pass
                    # print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}.")

                if new_state == Gst.State.PLAYING:
                    # print("Pipeline is now PLAYING.")
                    break
        else:
            pass
            # print("No bus message received. Retrying...")
    batch_count = 0
    while True:
        sample = sink.emit('pull-sample')
        # print('Pulled sample:', sample)
        if not sample:
            break
        buffer = sample.get_buffer()

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            # print(success)
            frame = np.frombuffer(map_info.data, np.uint8)
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value('width')
            height = caps.get_structure(0).get_value('height')
            expected_size = height * width * 3

            if frame.size == expected_size:
                frame = frame.reshape((height, width, 3))
                if frame.shape == frame_shape:
                    batch_buffer[batch_index] = frame
                    batch_index += 1

                    if batch_index == BATCH_SIZE:
                        # batch_queue.put(('new_batch', batch_index))
                        for control_queue in control_queues:
                            control_queue.put(('new_batch', batch_index, batch_count))
                        batch_index = 0
                        batch_count +=1
                else:
                    pass
                    # print(f"Frame shape {frame.shape} does not match expected {frame_shape}, skipping frame.")
            else:
                pass
                # print(f"Warning: Frame size mismatch. Expected {expected_size} elements, got {frame.size}")
            buffer.unmap(map_info)
        else:
            break

    # if batch_index > 0:
    #     batch_queue.put(('new_batch', batch_index))
    # print('ALL DONE')
    for control_queue in control_queues:
        control_queue.put(('stop', None, batch_count))
    # batch_queue.put(('stop', None))
    pipeline.set_state(Gst.State.NULL)
    shm.close()
    benchmark_result['end_time'] = time.time()
    benchmark_result['frame_count'] = frame_count


def gst_frame_consumer(shm_name, control_queues, result_queue, process_id):
    # print('Started consumer on main thread')

    facemesh_process = Process(target=process_batches_with_facemesh, args=(control_queues[0], shm_name, result_queue))
    mood_analysis_process = Process(target=process_batches_with_mood_analysis, args=(control_queues[1], shm_name, result_queue))
    # headpose_estimation = Process(target=process_batches_with_head_pose, args=(control_queues[2], shm_name, result_queue))

    facemesh_process.start()
    mood_analysis_process.start()
    # headpose_estimation.start()

    facemesh_process.join()
    mood_analysis_process.join()
    # headpose_estimation.join()

    # print('Finished all consumers')

