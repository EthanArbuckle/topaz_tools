import json
import subprocess
import time
from multiprocessing import Queue

from upscaler.common.models import FrameUpscalingJob, MediaUpscaleJob
from upscaler.workers.queue_worker import QueueWorker


class MediaUpscaleWorker(QueueWorker):
    def __init__(self, media_upscale_queue: Queue, frame_upscale_queue: Queue):
        super().__init__(queue=media_upscale_queue)
        self.upscale_jobs_queue = frame_upscale_queue

    def process_work(self, encoding_job: MediaUpscaleJob) -> None:

        if not encoding_job.source_media_path.exists():
            print(f"input file does not exist: {encoding_job.source_media_path}")
            return

        print(f"starting on {encoding_job.source_media_path}")
        encoding_job.job_start_time = time.time()

        png_output_path = encoding_job.png_output_path_root / f"topaz_{encoding_job.output_media_path.stem}"
        if not png_output_path.exists():
            png_output_path.mkdir()

        # Determine how many frames the media contains
        cmds = [
            "ffprobe",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            encoding_job.source_media_path.as_posix(),
        ]
        raw_probe_output = subprocess.check_output(cmds, stderr=subprocess.PIPE)
        probe_output = json.loads(raw_probe_output)
        if not probe_output or "streams" not in probe_output:
            raise RuntimeError(f"Failed to probe media: {probe_output}")

        # Topaz loosely guesses the framecount rather than doing a proper calculation, so this has to do the same inaccurate calculation.
        # Fetch duration (in seconds), rounded down to the nearest integer
        duration = int(float(probe_output["format"]["duration"]))
        # Topaz rounds framerate to 2 digit precision
        fps = round(eval(probe_output["streams"][0]["r_frame_rate"]), 2)
        # Guess framecount using duration * framerate, rounded down to the nearest integer
        total_frame_count = int(duration * fps)
        # Some AI models change the framerate. This needs to be tracked in order to accuralely resume a previous job, as the outputted
        # png file names don't necessarily correlate directly to a frame index from the original media
        print(
            f"total frames: orig:{total_frame_count} @ {fps}, outputting:{total_frame_count * encoding_job.ai_model.output_fps_increase} @ {fps * encoding_job.ai_model.output_fps_increase}"
        )

        frames = list(range(0, total_frame_count))
        frame_chunks = [frames[i : i + encoding_job.chunk_size] for i in range(0, len(frames), encoding_job.chunk_size)]

        for chunk_idx, chunk in enumerate(frame_chunks):
            chunk_png_output_path = png_output_path / str(chunk_idx)
            end_frame = chunk[-1] if chunk_idx < len(frame_chunks) - 1 else None
            self.upscale_jobs_queue.put(
                FrameUpscalingJob(
                    source_media_path=encoding_job.source_media_path,
                    output_media_path=encoding_job.output_media_path,
                    job_start_time=encoding_job.job_start_time,
                    png_output_path=chunk_png_output_path,
                    start_frame=chunk[0],
                    end_frame=end_frame,
                    ai_model=encoding_job.ai_model,
                    output_scale=encoding_job.output_scale,
                    total_chunk_count=len(frame_chunks),
                    output_fps=fps * encoding_job.ai_model.output_fps_increase,
                )
            )
