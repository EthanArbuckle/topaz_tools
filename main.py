import time
from multiprocessing import Queue
from pathlib import Path

from upscaler.common.models import AiModel, Gpu, MediaUpscaleJob
from upscaler.workers.completed_chunk_worker import CompletedChunkWorker
from upscaler.workers.frame_upscale_worker import FrameUpscaleWorker
from upscaler.workers.media_upscaler_worker import MediaUpscaleWorker
from upscaler.workers.png_encode_worker import PngBuilderWorker

AI_MODEL_DIONE_INTERLACED_ROBUST = AiModel("dtvs-1", 2)
AI_MODEL_DIONE_INTERLACED_TV_V3 = AiModel("dtv-3", 2)
AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12 = AiModel("amq-12", 1)

GPUS = [
    Gpu("Nvidia 980", 0, "media@192.168.1.181"),
    Gpu("Nvidia 980", 1, "media@192.168.1.181"),
    Gpu("Nvidia 980", 0),
    Gpu("AMD 5700xt", 1),
    Gpu("AMD 5700xt", 1),
    Gpu("AMD 5700xt", 1),
    Gpu("AMD 5700xt", 1),
]


def launch():
    media_upscale_queue: Queue = Queue()
    frame_upscale_queue: Queue = Queue()
    completed_chunk_queue: Queue = Queue()
    png_encode_queue: Queue = Queue()

    media_upscale_worker = MediaUpscaleWorker(media_upscale_queue, frame_upscale_queue)
    media_upscale_worker.start()

    for gpu in GPUS:
        frame_upscale_worker = FrameUpscaleWorker(frame_upscale_queue, completed_chunk_queue, gpu)
        frame_upscale_worker.start()

    completed_chunk_worker = CompletedChunkWorker(completed_chunk_queue, png_encode_queue)
    completed_chunk_worker.start()

    png_encode_worker = PngBuilderWorker(png_encode_queue)
    png_encode_worker.start()

    job = MediaUpscaleJob(
        source_media_path=Path("g:/Video/MONK/s01/disk04/title_t04.mkv"),
        output_media_path=Path("t:/monk_s01_extra_01.mkv"),
        png_output_path_root=Path("t:/"),
        ai_model=AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12,
        output_scale=2.25,
        chunk_size=250,
        job_start_time=0.0,  # TODO
    )
    media_upscale_queue.put(job)

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        exit(0)


if __name__ == "__main__":
    launch()
