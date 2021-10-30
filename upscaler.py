import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Optional, Tuple

from stream_indexer import determine_stream_indexes


@dataclass
class AiModel:
    short_name: str
    output_fps_increase: int


AI_MODEL_DIONE_INTERLACED_ROBUST = AiModel("dtvs-1", 2)
AI_MODEL_ARTEMIS_MEDIUM_QUALITY_V12 = AiModel("amq-12", 1)


@dataclass
class Gpu:
    name: str
    cuda_index: int
    work_allotment: float
    remote_host: Optional[str]

    def __str__(self) -> str:
        origin = "remote" if self.remote_host else "local"
        return f"[{origin} {self.name}]"


GPUS = [
    Gpu("Nvidia 980", 0, 0.20, "media@192.168.1.181"),
    Gpu("Nvidia 980", 1, 0.20, "media@192.168.1.181"),
    Gpu("Nvidia 980", 0, 0.22, None),
    Gpu("AMD 5700xt", 1, 0.38, None),
]


def _poll_progress(expected_frame_count: int, png_output_path: Path) -> None:

    previous_frame_count = 0
    completed_frame_count = 0
    while completed_frame_count != expected_frame_count:
        completed_frame_count = len([file.stem for file in png_output_path.iterdir() if "png" in file.suffix])
        completion_percentage = (100.0 / expected_frame_count) * completed_frame_count
        completion_percentage = round(completion_percentage, 2)

        rendering_fps = abs(completed_frame_count - previous_frame_count)
        if rendering_fps > 0:
            seconds_per_frame = round(1.0 / rendering_fps, 3)
            previous_frame_count = completed_frame_count

            eta = (expected_frame_count - completed_frame_count) * seconds_per_frame
            mm, ss = divmod(eta, 60)
            hh, mm = divmod(mm, 60)
            eta = "%d:%02d:%02d" % (hh, mm, ss)

            sys.stdout.write("\r")
            sys.stdout.write(
                f"progress: {completion_percentage}%, rendering at {rendering_fps}fps ({seconds_per_frame}s per frame) ETA: {eta}        "
            )
            sys.stdout.write("\r")
        time.sleep(1)


def _upscale_media_on_gpu(args: Tuple[Path, Path, AiModel, Gpu, int, Optional[int]]) -> None:
    media_path, png_output_dir, ai_model, gpu, gpu_start_frame, gpu_end_frame = args

    start_time = time.time()

    cmds = [
        "-i",
        media_path.as_posix(),
        # Output directory
        "-o",
        png_output_dir.as_posix(),
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

    proc = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True and proc.stderr:
        output = proc.stderr.readline()
        if not output:
            break
        # print(output.decode("utf-8"))
    proc.wait()

    mm, ss = divmod(time.time() - start_time, 60)
    hh, mm = divmod(mm, 60)
    s = "%d:%02d:%02d" % (hh, mm, ss)
    print(f"{gpu} finished in {hh}:{mm:02f}:{ss:02f}")


def _stitch_audio_from_original(original_media: Path, output_png_dir: Path, new_media_path: Path, fps: float) -> None:
    """Merge the audio and subtitles from the original media with the new upscaled pngs to produce the final media file"""

    start_time = time.time()

    # Figure out where the streams are located within the original track
    audio_stream, _, subtitle_stream = determine_stream_indexes(original_media)
    if not audio_stream:
        raise RuntimeError("Failed to find audio stream")

    cmds = [
        "ffmpeg",
        "-r",
        str(fps),
        "-f",
        # Input 1 is the pngs
        "image2",
        "-thread_queue_size",
        "512",
        "-i",
        f"{output_png_dir.as_posix()}/%06d.png",
        # Input 2 is the original media
        "-i",
        original_media.as_posix(),
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
        "18",
        "-pix_fmt",
        "yuv420p",
        new_media_path.as_posix(),
    ]

    subprocess.run(cmds)

    mm, ss = divmod(time.time() - start_time, 60)
    hh, mm = divmod(mm, 60)
    s = "%d:%02d:%02d" % (hh, mm, ss)
    print(f"finished encoding in {hh}:{mm:02f}:{ss:02f}")


def upscale_media(media_path: Path, output_path: Path, ai_model: AiModel) -> None:

    png_output_dir = Path(f"//DESKTOP-0TN2T9A/g/topaz_{output_path.stem}")
    existing_frames = []
    if not png_output_dir.exists():
        png_output_dir.mkdir()
    else:
        existing_frames = [file.stem for file in png_output_dir.iterdir() if "png" in file.suffix]

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
        f"total frames: orig:{total_frame_count} @ {fps}, outputting:{total_frame_count * ai_model.output_fps_increase} Existing frames: {len(existing_frames)}"
    )

    thread_work_args = []
    working_frame_index = 0
    planned_render_count = 0
    for gpu_index, gpu in enumerate(GPUS):
        # Determine how many frames this GPU will render
        gpu_frame_allotment_count = int(float(total_frame_count) * gpu.work_allotment)
        # Calculate the start and end frames this GPU will render
        gpu_start_frame = working_frame_index
        gpu_end_frame = gpu_start_frame + gpu_frame_allotment_count
        working_frame_index += gpu_frame_allotment_count

        # Handle existing frames
        adjusted_gpu_start_frame = gpu_start_frame
        for frame_index in range(gpu_start_frame, gpu_end_frame):
            padded_frame_index = str(frame_index * ai_model.output_fps_increase).zfill(6)
            if padded_frame_index not in existing_frames:
                break
            adjusted_gpu_start_frame += 1

        adjusted_gpu_frame_allotment_count = gpu_end_frame - adjusted_gpu_start_frame
        if adjusted_gpu_frame_allotment_count <= 0:
            # Already done
            print(f"skipping work for gpu. allotment: {adjusted_gpu_frame_allotment_count}")
            continue

        if adjusted_gpu_frame_allotment_count != gpu_frame_allotment_count:
            print(
                f"{gpu}: adjusting start from {gpu_start_frame} to {adjusted_gpu_start_frame}. Was {gpu_frame_allotment_count}, now getting {adjusted_gpu_frame_allotment_count}"
            )

        # The calculated framecount doesn't always match Topaz's (because they're inaccurate), and Topaz throws an error if endFrame > totalFrames.
        # To mitigate, the last GPU does not have an explicit end frame.
        # TODO: it'd be ideal if the "last gpu" was the most powerful in case the frame calculation was significantly off
        # and it's left with an unporportional amount of work.
        adjusted_gpu_end_frame: Optional[int] = gpu_end_frame
        if gpu_index == len(GPUS) - 1:
            adjusted_gpu_end_frame = None

        planned_render_count += adjusted_gpu_frame_allotment_count
        print(f"{gpu} getting {adjusted_gpu_frame_allotment_count} frames ({adjusted_gpu_start_frame}-{gpu_end_frame})")
        thread_work_args.append(
            (media_path, png_output_dir, ai_model, gpu, adjusted_gpu_start_frame, adjusted_gpu_end_frame)
        )
    print(f"going to render {planned_render_count}({planned_render_count * ai_model.output_fps_increase}) frames")

    # Start up a process that will print the overall progress of the work (across of gpus)
    p = Process(
        target=_poll_progress,
        args=(
            planned_render_count * ai_model.output_fps_increase,
            png_output_dir,
        ),
    )
    p.start()

    # Start upscaling
    with Pool(processes=len(thread_work_args)) as pool:
        pool.map(_upscale_media_on_gpu, thread_work_args)

    # Terminate the progress tracker if it has not already killed itself
    if p.is_alive():
        p.terminate()

    print("upscaling complete")

    # Build the pngs into the final media file
    _stitch_audio_from_original(media_path, png_output_dir, output_path, fps * ai_model.output_fps_increase)


if __name__ == "__main__":

    input_path = Path("//DESKTOP-0TN2T9A/g/Video/MONK/s01/disk02/title_t00.mkv")
    if not input_path.exists():
        raise RuntimeError("Failed to find input file")

    output_path = Path("g:/monk_s01e04.mkv")
    if output_path.exists():
        shutil.rmtree(output_path.as_posix(), ignore_errors=True)

    upscale_media(input_path, output_path, ai_model=AI_MODEL_DIONE_INTERLACED_ROBUST)