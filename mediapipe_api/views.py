from celery.result import AsyncResult
from .tasks import process_video_task
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import json
from gi.repository import Gst
import gi
import threading
import numpy as np
from queue import Queue
import os
import cv2
import matplotlib.pyplot as plt
from multiprocessing import shared_memory, Process, Queue, Manager
# from .utils import producer, consumer
from .utils import gst_frame_consumer, gst_frame_producer


os.environ['GST_GL_API'] = 'disabled'
gi.require_version('Gst', '1.0')
Gst.init(None)

manager = Manager()


# Create your views here.

# class ProcessVideoAPIView(APIView):

#     def post(self, request, *args, **kwargs):
#         # url = json.loads(request.body)['url']
#         # # stream_video(url)
#         # queue_size = 10
#         # frame_queue = Queue(maxsize=queue_size)
#         # benchmark_result = {
#         #     'start_time': None,
#         #     'end_time': None,
#         #     'frame_count': 0
#         # }
#         # # setup pipeline
#         # gst_pipeline_str = (
#         #     f'souphttpsrc location={url} ! queue ! matroskademux ! queue ! vp9dec ! queue ! videoconvert ! queue !'
#         #     f'video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=sink max-buffers=30 drop=true'
#         # )

#         # pipeline = Gst.parse_launch(gst_pipeline_str)
#         # appsink = pipeline.get_by_name('sink')
#         # appsink.set_property('emit-signals', True)
#         # appsink.set_property('sync', False)
        

#         # producer_thread = threading.Thread(
#         #     target=producer, args=(pipeline, appsink, frame_queue, benchmark_result))
#         # producer_thread.start()

#         # # consumer(frame_queue)
#         # consumer_thread = threading.Thread(
#         #     target=consumer, args=(frame_queue,))
#         # consumer_thread.start()


#         # producer_thread.join()
#         # consumer_thread.join()
        
#         # # Calculate benchmark
#         # total_time = benchmark_result['end_time'] - \
#         #     benchmark_result['start_time']
#         # fps = benchmark_result['frame_count'] / \
#         #     total_time if total_time > 0 else 0

#         # print('Finished Streaming')
#         # print(f'Total Frames: {benchmark_result["frame_count"]}')
#         # print(f'Total Time: {total_time:.2f} seconds')
#         # print(f'Processing Speed: {fps:.2f} FPS')

#         # return Response({
#         #     "url": url,
#         #     "frames_processed": benchmark_result["frame_count"],
#         #     "total_time_seconds": total_time,
#         #     "processing_fps": fps
#         # }, status=200)




#         url = json.loads(request.body)['url']


#         benchmark_result = manager.dict({
#             'start_time': None,
#             'end_time': None,
#             'frame_count': 0
#         })
#         result_queue = manager.dict({
#             "eye_out_of_view":[],
#             "overall_mood":""
#         })
#         frame_shape = (720,1280,3)
#         batch_size = 5
#         total_size = np.prod((batch_size,) + frame_shape)*np.dtype(np.uint8).itemsize

#         shm = shared_memory.SharedMemory(create=True, size=total_size)
#         control_queue_facemesh = Queue()
#         control_queue_mood = Queue()
#         control_queue_headpose = Queue()

#         control_queues = [control_queue_facemesh,
#                           control_queue_mood]
        
#         # import gi
#         # gi.require_version('Gst', '1.0')
#         # from gi.repository import Gst

#         # Gst.init(None)
#         # # setup pipeline
#         # gst_pipeline_str = (
#         #     f'souphttpsrc location={url} ! queue ! matroskademux ! queue ! vp9dec ! queue ! videoconvert ! queue !'
#         #     f'video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=sink max-buffers=30 drop=true'
#         # )

#         producer_process = Process(target=gst_frame_producer, args=(url, benchmark_result, shm.name, frame_shape, control_queues, batch_size))
#         consumer_process = Process(target=gst_frame_consumer, args=(shm.name, control_queues, result_queue, 1))

#         producer_process.start()
#         consumer_process.start()


#         producer_process.join()
#         if producer_process.is_alive():
#             print("Producer still running, terminating...")
#             producer_process.terminate()

#         consumer_process.join()
#         if consumer_process.is_alive():
#             print("Consumer still running, terminating...")
#             consumer_process.terminate()


#         print("Producer exit code:", producer_process.exitcode)
#         print("Consumer exit code:", consumer_process.exitcode)

#         try:
#             shm.close()
#             shm.unlink()
#         except Exception as e:
#             print(f"Error during shm cleanup: {e}")

#         from collections import defaultdict

#         moods = []
#         if "mood_analysis" in result_queue:
#             for item in result_queue["mood_analysis"]:
#                 moods.append(item)

#         sentiment_by_timeblock = defaultdict(list)
#         for entry in moods:
#             block = int(entry['video_timestamp'] // 5)
#             sentiment_by_timeblock[block].append(entry['mood_score'])

#         sentiment_summary = {}
#         for block, scores in sentiment_by_timeblock.items():
#             avg_score = sum(scores) / len(scores)
#             sentiment_summary[f"{block * 5}-{(block + 1) * 5}s"] = round(avg_score, 2)

#         result_queue['sentiment_over_time'] = sentiment_summary
#         del result_queue['mood_analysis']
            
#         # Calculate benchmark
#         start_time = benchmark_result.get('start_time')
#         end_time = benchmark_result.get('end_time')
#         total_time = (end_time - start_time) if start_time is not None and end_time is not None else 0
#         fps = benchmark_result['frame_count'] / total_time if total_time > 0 else 0

#         print('Finished Streaming')
#         print(f'Total Frames: {benchmark_result["frame_count"]}')
#         print(f'Total Time: {total_time:.2f} seconds')
#         print(f'Processing Speed: {fps:.2f} FPS')

#         return Response({
#             "url": url,
#             "frames_processed": benchmark_result["frame_count"],
#             "total_time_seconds": total_time,
#             "processing_fps": fps,
#             "results":result_queue,
#         }, status=200)


class ProcessVideoAPIView(APIView):
    def post(self, request, *args, **kwargs):
        url = json.loads(request.body)['url']
        task = process_video_task.delay(url)
        return Response({
            'task_id':task.id,
            'status':'Processing',
            'message': f"Poll /api/v1/mediapipe/status/{task.id}/ for result."
        }, status=202)


class TaskStatusAPIView(APIView):
    def get(self, request, task_id):
        task = AsyncResult(str(task_id))
        if task.state == 'PENDING':
            return Response({"status": "pending"}, status=202)
        elif task.state == 'SUCCESS':
            return Response({"status": "completed", "result": task.result})
        elif task.state == 'FAILURE':
            return Response({"status": "failed", "error": str(task.result)}, status=500)
        else:
            return Response({"status": task.state}, status=202)
