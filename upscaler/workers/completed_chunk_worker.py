import shutil
from multiprocessing import Queue
from pathlib import Path
from typing import Dict, List

from upscaler.common.models import CompletedChunk, PngEncodingJob
from upscaler.workers.queue_worker import QueueWorker


class CompletedChunkWorker(QueueWorker):
    def __init__(self, completed_chunk_queue: Queue, png_encode_queue: Queue):
        super().__init__(queue=completed_chunk_queue)
        self.png_encode_queue = png_encode_queue
        self.chunks: Dict[Path, List[CompletedChunk]] = {}

    def process_work(self, completed_chunk: CompletedChunk) -> None:

        if completed_chunk.source_media_path not in self.chunks:
            self.chunks[completed_chunk.source_media_path] = []

        chunk_collection = self.chunks[completed_chunk.source_media_path]
        chunk_collection.append(completed_chunk)

        print(f"recieved chunk {len(chunk_collection)} of {completed_chunk.total_chunk_count}")
        if completed_chunk.total_chunk_count == len(chunk_collection):
            # Move png chunks into the root png directory
            root_png_output_path = completed_chunk.chunk_png_output_path.parent
            for chunk_dir in root_png_output_path.iterdir():
                if not chunk_dir.is_dir():
                    continue
                for png_frame in chunk_dir.glob("*.png"):
                    try:
                        new_frame_path = root_png_output_path / png_frame.name
                        shutil.move(png_frame.as_posix(), new_frame_path.as_posix())
                    except shutil.Error as e:
                        print(f"failed to move file {png_frame.as_posix()}: {e}")

            # Start building the final output file
            self.png_encode_queue.put(
                PngEncodingJob(
                    source_media_path=completed_chunk.source_media_path,
                    output_media_path=completed_chunk.output_media_path,
                    job_start_time=completed_chunk.job_start_time,
                    png_output_path_root=root_png_output_path,
                    output_fps=completed_chunk.output_fps,
                )
            )
