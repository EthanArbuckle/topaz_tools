import queue
import subprocess
import time
from multiprocessing import Queue
from pathlib import Path
from threading import Event, Thread
from dataclasses import dataclass
from stream_indexer import determine_stream_indexes
import os


@dataclass
class PngEncodingJob:
    original_media_path: Path
    output_png_dir: Path
    new_media_path: Path
    output_fps: float


class QueueWorker(Thread):
    def __init__(self, queue=None):
        Thread.__init__(self)

        self.daemon = True
        self.exit = Event()

        if not queue:
            queue = Queue()
        self.queue = queue

    def run(self) -> None:
        while not self.exit.is_set():
            try:
                data = self.queue.get()
                self.process_work(data)
            except Exception as e:
                print(e)
                continue


    def process_work(self, encoding_job):
        pass


class PngBuilderWorker(QueueWorker):
    def process_work(self, encoding_job: PngEncodingJob) -> None:
        """Merge the audio and subtitles from the original media with the new upscaled pngs to produce the final media file"""

        start_time = time.time()

        # Figure out where the streams are located within the original track
        audio_stream, _, subtitle_stream = determine_stream_indexes(encoding_job.original_media_path)
        if not audio_stream:
            raise RuntimeError("Failed to find audio stream")

        cmds = [
            "ffmpeg",
            "-f",
            # Input 1 is the pngs
            "image2",
            "-thread_queue_size",
            str(8192 * 4),
            "-r",
            str(encoding_job.output_fps),
            "-i",
            f"{encoding_job.output_png_dir.as_posix()}/%06d.png",
            # Input 2 is the original media
            "-i",
            encoding_job.original_media_path.as_posix(),
            # Map the video stream from input1
            "-map",
            "0:0",
            # Map the audio from the original media
            "-map",
            f"1:{audio_stream[-1]}",
        ]

        # Map subtitles if they were found in the original media
        if subtitle_stream:
            cmds += ["-map", f"1:{subtitle_stream[-1]}", "-scodec", "copy"]

        cmds += [
            # h264 encode the video output.
            # Use AMD hardware encoding
            "-vcodec",
            "hevc_amf",
            # Preserve audio codec (but maybe it should be ac3 for ps4?)
            "-acodec",
            "copy",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv420p",
            encoding_job.new_media_path.as_posix(),
        ]

        try:
            subprocess.run(cmds, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            mm, ss = divmod(time.time() - start_time, 60)
            hh, mm = divmod(mm, 60)
            duration = "%d:%02d:%02d" % (hh, mm, ss)
            print(f"finished encoding in {duration}")

            if encoding_job.new_media_path.exists():
                start_time = time.time()

                cmd_path = os.path.join(
                    os.environ["SYSTEMROOT"] if "SYSTEMROOT" in os.environ else r"C:\Windows", "System32", "cmd.exe"
                )
                delete_path = os.path.abspath(encoding_job.output_png_dir.as_posix()).replace("/", "\\")
                args = [cmd_path, "/C", "rmdir", "/S", "/Q", f"{delete_path}"]
                print(args)
                subprocess.run(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

                mm, ss = divmod(time.time() - start_time, 60)
                hh, mm = divmod(mm, 60)
                duration = "%d:%02d:%02d" % (hh, mm, ss)
                print(f"finished deleting pngs in {duration}")
        except Exception as e:
            print(e)
            pass
