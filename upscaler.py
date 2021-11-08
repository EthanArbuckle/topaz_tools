import json
import shutil
import subprocess
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing.dummy import Pool
from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple

from stream_indexer import determine_stream_indexes


@dataclass
class AiModel:
    short_name: str
    output_fps_increase: int


@dataclass
class UpscalingJob:
    source_file: str
    destination_file: str
    png_output_dir: str
    ai_model: AiModel


AI_MODEL_DIONE_INTERLACED_ROBUST = AiModel("dtvs-1", 2)
AI_MODEL_DIONE_INTERLACED_TV_V3 = AiModel("dtv-3", 2)
AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12 = AiModel("amq-12", 1)
LOCAL_GPU = None


@dataclass
class Gpu:
    name: str
    cuda_index: int
    work_allotment: float
    remote_host: Optional[str]
    png_output_dir: Optional[Path] = None
    expected_frame_render_count: int = 0
    rough_expected_frame_render_count: int = 0
    active: bool = False

    def __str__(self) -> str:
        origin = "remote" if self.remote_host else "local"
        return f"[{origin} {self.name}]"


@dataclass
class GpuJobProgress:
    gpu_name: str
    expected_frame_count: int
    png_output_path: Path
    completed_frame_count = 0
    last_pass_completed_frame_count = 0
    rendering_fps = 0.0
    rendering_spf = 0.0
    remaining_rendering_time_seconds = 0.0

    def __str__(self) -> str:
        rounded_fps = round(self.rendering_fps, 2)
        return f"{self.gpu_name} {rounded_fps}fps"


def _poll_progress(gpus: List[Gpu]) -> None:
    gpu_progress_jobs = [
        GpuJobProgress(str(gpu), gpu.expected_frame_render_count, gpu.png_output_dir)
        for gpu in gpus
        if gpu.png_output_dir
    ]

    total_expected_frames = sum([gpu.expected_frame_render_count for gpu in gpus])
    total_completed_frames = 0
    last_pass_total_completed_frames = 0

    time.sleep(1)
    while total_completed_frames < total_expected_frames and any([gpu.active for gpu in gpus]):
        for gpu in gpu_progress_jobs:

            gpu.completed_frame_count = len(
                [file.stem for file in gpu.png_output_path.iterdir() if "png" in file.suffix]
            )

            gpu.rendering_fps = gpu.completed_frame_count - gpu.last_pass_completed_frame_count
            if gpu.rendering_fps > 0:
                gpu.rendering_spf = round(1.0 / gpu.rendering_fps, 3)
                gpu.last_pass_completed_frame_count = gpu.completed_frame_count

                gpu.remaining_rendering_time_seconds = (
                    gpu.expected_frame_count - gpu.completed_frame_count
                ) * gpu.rendering_spf

        total_completed_frames = sum([gpu.completed_frame_count for gpu in gpu_progress_jobs])
        largest_eta = sorted([gpu.remaining_rendering_time_seconds for gpu in gpu_progress_jobs], reverse=True)[0]
        overall_render_fps = abs(total_completed_frames - last_pass_total_completed_frames)
        overall_render_spf = 0.0
        if overall_render_fps > 0:
            overall_render_spf = round(1.0 / overall_render_fps, 3)

        mm, ss = divmod(largest_eta, 60)
        hh, mm = divmod(mm, 60)
        formatted_eta = "%d:%02d:%02d" % (hh, mm, ss)

        completion_percentage = (100.0 / total_expected_frames) * total_completed_frames
        completion_percentage = round(completion_percentage, 2)

        last_pass_total_completed_frames = total_completed_frames

        gpus_progress_string = ", ".join(
            [str(gpu) for gpu in gpu_progress_jobs if gpu.completed_frame_count < gpu.expected_frame_count]
        )
        sys.stdout.write("\r")
        sys.stdout.write(
            f"{completion_percentage}%, {overall_render_fps}fps {overall_render_spf}spf ETA: {formatted_eta}. {gpus_progress_string}    "
        )
        sys.stdout.write("\r")
        time.sleep(1)


def _upscale_media_on_gpu(args: Tuple[Path, AiModel, Gpu, int, Optional[int]]) -> None:
    media_path, ai_model, gpu, gpu_start_frame, gpu_end_frame = args

    if not gpu.png_output_dir or not gpu.png_output_dir.exists():
        raise RuntimeError(f"Invalid png output path for {gpu}")

    start_time = time.time()
    gpu.active = True

    cmds = [
        "-i",
        media_path.as_posix(),
        # Output directory
        "-o",
        gpu.png_output_dir.as_posix(),
        # Save output as lossless png files
        "-f",
        "png",
        # Upscale 225%
        "-s",
        "2.25",
        # The AI model to use
        "-m",
        ai_model.short_name,
        # Grain
        "-a",
        "1.8",
        # Which GPU to use
        "-c",
        str(gpu.cuda_index),
        # Start frame
        "-b",
        str(gpu_start_frame),
    ]

    # Explicit end frame, if provided
    if gpu_end_frame:
        cmds += [
            "-e",
            str(gpu_end_frame),
        ]

    # If this is a remote GPU, invoke the command over ssh
    if gpu.remote_host:
        cmds = [
            "ssh",
            gpu.remote_host,
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
    mm, ss = divmod(time.time() - start_time, 60)
    hh, mm = divmod(mm, 60)
    duration = "%d:%02d:%02d" % (hh, mm, ss)
    print(f"{gpu} finished in {duration}")

    gpu.active = False


def upscale_media(
    media_path: Path, output_path: Path, ai_model: AiModel, upscaled_png_dir: Path, gpus: List[Gpu]
) -> None:

    existing_frames = []
    if not upscaled_png_dir.exists():
        upscaled_png_dir.mkdir()
    else:
        existing_frames = [file.stem for file in upscaled_png_dir.iterdir() if "png" in file.suffix]

    # Determine how many frames the media contains
    cmds = [
        "ffprobe",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        media_path.as_posix(),
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
        f"total frames: orig:{total_frame_count} @ {fps}, outputting:{total_frame_count * ai_model.output_fps_increase} @ {fps * ai_model.output_fps_increase}. Existing frames: {len(existing_frames)}"
    )

    thread_work_args = []
    working_frame_index = 0
    planned_render_count = 0
    for gpu_index, gpu in enumerate(gpus):

        # Create the output dir for the GPUs rendered frames.
        # Each GPU has its own folder to enable tracking the render speed of each one
        sanitized_gpu_name = gpu.name.replace(" ", "_")
        gpu.png_output_dir = upscaled_png_dir / f"{gpu_index}_{sanitized_gpu_name}"
        if not gpu.png_output_dir.exists():
            gpu.png_output_dir.mkdir()

        # Calculate the start and end frames this GPU will render
        gpu_frame_allotment_count = int(float(total_frame_count) * gpu.work_allotment)
        gpu_start_frame = working_frame_index
        gpu_end_frame = gpu_start_frame + gpu_frame_allotment_count
        working_frame_index += gpu_frame_allotment_count

        # Handle existing frames
        existing_frames = sorted(
            [file.stem for file in gpu.png_output_dir.iterdir() if "png" in file.suffix], reverse=True
        )
        if len(existing_frames) > 0:
            last_rendered_frame = int(existing_frames[0]) / ai_model.output_fps_increase
            if last_rendered_frame > gpu_end_frame:
                print(f"{gpu} output directory already contains all the expected frames")
                continue
            # Move the starting frame up to right before the last found frame on disk.
            # The overlap is in case the last rendered png is malformed due to a topaz crash
            gpu_start_frame = int(last_rendered_frame) - 2
            gpu_frame_allotment_count = gpu_end_frame - gpu_start_frame

        gpu.expected_frame_render_count = gpu_frame_allotment_count * ai_model.output_fps_increase

        # The calculated framecount doesn't always match Topaz's (because they're inaccurate), and Topaz throws an error if endFrame > totalFrames.
        # To mitigate, the last GPU does not have an explicit end frame.
        # TODO: it'd be ideal if the "last gpu" was the most powerful in case the frame calculation was significantly off
        # and it's left with an unporportional amount of work.
        adjusted_gpu_end_frame: Optional[int] = gpu_end_frame
        if gpu_index == len(gpus) - 1:
            adjusted_gpu_end_frame = None
            gpu.expected_frame_render_count = total_frame_count - gpu_start_frame

        planned_render_count += gpu_frame_allotment_count
        print(f"{gpu} getting {gpu.expected_frame_render_count} frames ({gpu_start_frame}-{gpu_end_frame})")

        thread_work_args.append((media_path, ai_model, gpu, gpu_start_frame, adjusted_gpu_end_frame))
    print(f"going to render {planned_render_count}({planned_render_count * ai_model.output_fps_increase}) frames")

    # Start upscaling
    if thread_work_args:
        # Start up a process that will print the overall progress of the work (across of gpus)
        p = Thread(
            target=_poll_progress,
            args=(gpus,),
        )
        p.daemon = True
        p.start()

        with Pool(processes=len(thread_work_args)) as pool:
            pool.map(_upscale_media_on_gpu, thread_work_args)
        print("upscaling complete")

    # Verify the frames were created as expected
    for gpu in gpus:

        rendered_frames = []
        if gpu.png_output_dir and gpu.png_output_dir.exists():
            rendered_frames = sorted(
                [file.stem for file in gpu.png_output_dir.iterdir() if "png" in file.suffix], reverse=True
            )
        if gpu.expected_frame_render_count > len(rendered_frames):
            print(
                f"GPU did not render the expected amount of frames. Expected {gpu.expected_frame_render_count} but found {len(rendered_frames)}"
            )
            raise RuntimeError()

    # Merge all the pngs together for ffmpeg
    print(f"moving pngs...")
    start_time = time.time()
    for gpu in gpus:
        if not gpu.png_output_dir:
            continue

        for frame in gpu.png_output_dir.glob("*.png"):
            try:
                new_frame_path = upscaled_png_dir / frame.name
                shutil.move(frame.as_posix(), new_frame_path.as_posix())
            except shutil.Error as e:
                print(f"failed to move file {frame.as_posix()}: {e}")

    mm, ss = divmod(time.time() - start_time, 60)
    hh, mm = divmod(mm, 60)
    duration = "%d:%02d:%02d" % (hh, mm, ss)
    print(f"finished moving pngs in {duration}")


def perform_job(upscaling_job: UpscalingJob, gpus: List[Gpu]) -> None:
    input_path = Path(upscaling_job.source_file)
    if not input_path.exists():
        raise RuntimeError("Failed to find input file")

    print(f"starting on: {input_path.as_posix()}")
    output_path = Path(upscaling_job.destination_file)
    upscaling_job.png_output_dir = Path(upscaling_job.png_output_dir) / f"topaz_{output_path.stem}"
    upscale_media(input_path, output_path, upscaling_job.ai_model, upscaling_job.png_output_dir, gpus)
