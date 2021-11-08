from multiprocessing import Lock, Queue
from pathlib import Path

from upscaler import (
    AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12,
    AI_MODEL_DIONE_INTERLACED_TV_V3,
    LOCAL_GPU,
    Gpu,
    UpscalingJob,
    perform_job,
)
from worker import PngBuilderWorker, PngEncodingJob

encoder_work_queue = Queue()
encoder = PngBuilderWorker(queue=encoder_work_queue)
encoder.start()


if __name__ == "__main__":

    GPUS = [
        Gpu("Nvidia 980", 0, 0.12, "media@192.168.1.181"),
        Gpu("Nvidia 980", 1, 0.12, "media@192.168.1.181"),
        Gpu("Nvidia 980", 0, 0.12, LOCAL_GPU),
        Gpu("AMD 5700xt", 1, 0.16, LOCAL_GPU),
        Gpu("AMD 5700xt", 1, 0.16, LOCAL_GPU),
        Gpu("AMD 5700xt", 1, 0.16, LOCAL_GPU),
        Gpu("AMD 5700xt", 1, 0.16, LOCAL_GPU),
        # Gpu("Nvidia 2080TI", 0, 0.08, "topaz@192.168.1.137"),
        # Gpu("Nvidia 2080TI", 0, 0.07, "topaz@192.168.1.137"),
        # Gpu("Nvidia 2080TI", 0, 0.07, "topaz@192.168.1.137"),
        # Gpu("Nvidia 2080TI", 0, 0.07, "topaz@192.168.1.137"),
    ]
    # GPUS = [
    #     Gpu("Nvidia 980", 0, 0.16, LOCAL_GPU),
    #     Gpu("AMD 5700xt", 1, 0.21, LOCAL_GPU),
    #     Gpu("AMD 5700xt", 1, 0.21, LOCAL_GPU),
    #     Gpu("AMD 5700xt", 1, 0.21, LOCAL_GPU),
    #     Gpu("AMD 5700xt", 1, 0.22, LOCAL_GPU),
    # ]
    total_work_allotments = sum([gpu.work_allotment for gpu in GPUS])
    print(total_work_allotments)
    if total_work_allotments < 1.0:
        raise RuntimeError("Bad work distribution")

    jobs = [
        UpscalingJob(
            "g:/Video/MONK/s04/disk02/title_t02.mkv",
            "g:/monk_test.mkv",
            "g:/",
            AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12,
        ),
        # UpscalingJob(
        #     "g:/iPod2.mp4",
        #     "t:/ipod_upscaled.mkv",
        #     "t:/",
        #     AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12,
        # ),
        # UpscalingJob(
        #     "g:/iPod2.mp4",
        #     "t:/ipod_upscaled_2.mkv",
        #     "t:/",
        #     AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12,
        # ),
    ]

    for job in jobs:

        stop_file = Path("stop.txt")
        if stop_file.exists():
            print("stopping early because stop.txt")
            break

        try:
            perform_job(job, GPUS)
            encoder_work_queue.put(
                PngEncodingJob(
                    original_media_path=Path(job.source_file),
                    output_png_dir=job.png_output_dir,
                    new_media_path=Path(job.destination_file),
                    output_fps=29.97 * job.ai_model.output_fps_increase,
                )
            )
        except Exception as e:
            print(f"failed to upscale {job.source_file}: {e}")

import time
try:
    while True:
        time.sleep(1)
except Exception as e:
    exit(0)
