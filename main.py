import time
from multiprocessing import Queue
from pathlib import Path

from upscaler.common.models import AiModel, Gpu, MediaUpscaleJob
from upscaler.workers.completed_chunk_worker import CompletedChunkWorker
from upscaler.workers.frame_upscale_worker import FrameUpscaleWorker
from upscaler.workers.media_upscaler_worker import MediaUpscaleWorker
from upscaler.workers.png_encode_worker import PngBuilderWorker
from upscaler.workers.progress_reporter_worker import ProgressReporterWorker


AI_MODEL_DIONE_INTERLACED_ROBUST = AiModel("dtvs-1", 2)
AI_MODEL_DIONE_INTERLACED_TV_V3 = AiModel("dtv-3", 2)
AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12 = AiModel("amq-12", 1)

GPUS = [
    Gpu("Nvidia 980-1", 0, "media@192.168.1.181"),
    Gpu("Nvidia 980-2", 1, "media@192.168.1.181"),
    Gpu("Nvidia 980-3", 0),
    Gpu("AMD 5700xt-1", 1),
    Gpu("AMD 5700xt-2", 1),
    Gpu("AMD 5700xt-3", 1),
    Gpu("AMD 5700xt-4", 1),
]


def launch():
    media_upscale_queue: Queue = Queue()
    frame_upscale_queue: Queue = Queue()
    completed_chunk_queue: Queue = Queue()
    png_encode_queue: Queue = Queue()
    render_speed_queue = Queue()

    media_upscale_worker = MediaUpscaleWorker(media_upscale_queue, frame_upscale_queue)
    media_upscale_worker.start()

    for gpu in GPUS:
        frame_upscale_worker = FrameUpscaleWorker(frame_upscale_queue, completed_chunk_queue, render_speed_queue, gpu)
        frame_upscale_worker.start()

    completed_chunk_worker = CompletedChunkWorker(completed_chunk_queue, png_encode_queue)
    completed_chunk_worker.start()

    progress_reporter_worker = ProgressReporterWorker(render_speed_queue)
    progress_reporter_worker.start()

    png_encode_worker = PngBuilderWorker(png_encode_queue)
    png_encode_worker.start()

    for media_file in Path("g:/Video/MONK/s05").glob("*/*.mkv"):
        print(f"in: {media_file}")
        out_name = f"{media_file.parts[3]}_{media_file.parts[4]}_{media_file.parts[5]}"
        outpath = Path("g:/monk_hd/") / out_name
        job = MediaUpscaleJob(
            source_media_path=media_file,
            output_media_path=outpath,
            png_output_path_root=Path("t:/"),
            ai_model=AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12,
            output_scale=2.25,
            chunk_size=1000,
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
