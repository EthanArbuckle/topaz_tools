import sys
from multiprocessing import Queue
from typing import Dict, Tuple

from upscaler.common.models import Gpu, GpuProgressEvent, ProgressEvent
from upscaler.workers.queue_worker import QueueWorker


class ProgressReporterWorker(QueueWorker):
    def __init__(self, progress_event_queue: Queue):
        super().__init__(queue=progress_event_queue)
        self.gpu_render_rates: Dict[Gpu, float] = {}
        self.gpu_frame_progress: Dict[Gpu, Tuple[int, int]] = {}

    def process_work(self, progress_event: ProgressEvent) -> None:
        if isinstance(progress_event, GpuProgressEvent):

            self.gpu_render_rates[progress_event.gpu] = progress_event.fps
            self.gpu_frame_progress[progress_event.gpu] = (
                progress_event.completed_frames,
                progress_event.expected_frames,
            )

            total_fps = sum(self.gpu_render_rates.values())
            total_rendered_frames = sum([rendered_frames for rendered_frames, _ in self.gpu_frame_progress.values()])
            total_expected_frames = sum([expected_frames for _, expected_frames in self.gpu_frame_progress.values()])
            progress_percent = round((100.0 / total_expected_frames) * total_rendered_frames, 2)

            sys.stdout.write("\r")
            sys.stdout.write(f"Rendering at {total_fps}fps - {progress_percent}%\t")
            sys.stdout.write("\r")
