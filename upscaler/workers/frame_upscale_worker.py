import subprocess
from multiprocessing import Queue

from upscaler.common.models import CompletedChunk, FrameUpscalingJob, Gpu
from upscaler.workers.queue_worker import QueueWorker


class FrameUpscaleWorker(QueueWorker):
    def __init__(self, frame_upscale_queue: Queue, completed_chunk_queue: Queue, gpu: Gpu):
        super().__init__(queue=frame_upscale_queue)
        self.gpu = gpu
        self.completed_chunk_queue = completed_chunk_queue

    def process_work(self, encoding_job: FrameUpscalingJob) -> None:

        if not encoding_job.png_output_path.exists():
            encoding_job.png_output_path.mkdir()

        print(f"{self.gpu} starting on chunk {encoding_job.start_frame}-{encoding_job.end_frame}")

        cmds = [
            "-i",
            encoding_job.source_media_path.as_posix(),
            # Output directory
            "-o",
            encoding_job.png_output_path.as_posix(),
            # Save output as lossless png files
            "-f",
            "png",
            # Scale
            "-s",
            str(encoding_job.output_scale),
            # The AI model to use
            "-m",
            encoding_job.ai_model.short_name,
            # Grain
            "-a",
            "1.8",
            # Which GPU to use
            "-c",
            str(self.gpu.cuda_index),
            # Start frame
            "-b",
            str(encoding_job.start_frame),
        ]

        # Explicit end frame, if provided
        if encoding_job.end_frame:
            cmds += [
                "-e",
                str(encoding_job.end_frame),
            ]

        # If this is a remote GPU, invoke the command over ssh
        if self.gpu.remote_host:
            cmds = [
                "ssh",
                self.gpu.remote_host,
                '"C:/Program Files/Topaz Labs LLC/Topaz Video Enhance AI 2.3/veai.exe"',  # this path has to be wrapped in double-quotes to satisfy ssh
            ] + cmds
        else:
            cmds = ["C:/Program Files/Topaz Labs LLC/Topaz Video Enhance AI 2.3/veai.exe"] + cmds

        proc = subprocess.Popen(cmds, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        # while True and proc.stderr:
        #     output = proc.stderr.readline()
        #     if not output:
        #         break
        # print(output.decode("utf-8"))
        proc.wait()

        self.completed_chunk_queue.put(
            CompletedChunk(
                source_media_path=encoding_job.source_media_path,
                output_media_path=encoding_job.output_media_path,
                chunk_png_output_path=encoding_job.png_output_path,
                total_chunk_count=encoding_job.total_chunk_count,
                output_fps=encoding_job.output_fps,
            )
        )
