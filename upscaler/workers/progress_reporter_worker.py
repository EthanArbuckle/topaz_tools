import sys
from multiprocessing import Queue
from typing import Dict, Tuple

from upscaler.common.models import Gpu, GpuProgressEvent, ProgressEvent
from upscaler.workers.queue_worker import QueueWorker


class ProgressReporterWorker(QueueWorker):
    def __init__(self, progress_event_queue: Queue):
        super().__init__(queue=progress_event_queue)
        self.gpu_last_progress_update: Dict[Gpu, ProgressEvent] = {}

    def process_work(self, progress_event: ProgressEvent) -> None:
        if isinstance(progress_event, GpuProgressEvent):

            self.gpu_last_progress_update[progress_event.gpu] = progress_event

            total_fps = sum([progress_event.fps for progress_event in self.gpu_last_progress_update.values()])
            total_rendered_frames = sum(
                [progress_event.completed_frames for progress_event in self.gpu_last_progress_update.values()]
            )
            total_expected_frames = sum(
                [progress_event.expected_frames for progress_event in self.gpu_last_progress_update.values()]
            )
            # progress_percent = round((100.0 / total_expected_frames) * total_rendered_frames, 2)

            sorted_gpus = sorted(self.gpu_last_progress_update.values(), key=lambda x: x.gpu.name)
            gpu_job_descriptions = ", ".join(
                [f"[{progress_event.gpu.name}=={progress_event.fps}fps]" for progress_event in sorted_gpus]
            )

            sys.stdout.write("\r")
            sys.stdout.write(f"{total_fps}fps | {gpu_job_descriptions}         ")
            sys.stdout.write("\r")
