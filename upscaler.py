import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Optional, Tuple


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
        return f"[({origin}) {self.name}]"


GPUS = [
    Gpu("Nvidia 980", 0, 0.20, "media@192.168.1.181"),
    Gpu("Nvidia 980", 1, 0.20, "media@192.168.1.181"),
    Gpu("Nvidia 980", 0, 0.22, None),
    Gpu("AMD 5700xt", 1, 0.38, None),
]


def _stream_index_from_ffmpeg_output(output: bytes) -> str:
    """Get the stream index from a ffpmeg output line"""
    # input: Stream #0:2(und): Video: h264
    # output: 0:2
    return output.split(b"#")[1].split(b"(")[0].decode("utf-8")[0:3]


def determine_stream_indexes(file_location: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Given a media file, locate the indexes of the individual streams"""
    audio_stream, video_stream, subtitle_stream = None, None, None
    try:
        # Print information about the provided media file
        subprocess.check_output(["ffmpeg.exe", "-i", file_location.as_posix()], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        for ffpmeg_output in error.stderr.splitlines():
            if b"Stream" not in ffpmeg_output:
                # We're only interested in info about the track's streams
                continue
            # A stream was found, carve out the index (for map operation)
            stream_index = _stream_index_from_ffmpeg_output(ffpmeg_output)
            # Identify what kind of stream this is
            if b"Subtitle" in ffpmeg_output:
                subtitle_stream = stream_index
            elif b"Video" in ffpmeg_output:
                video_stream = stream_index
            elif b"Audio" in ffpmeg_output:
                audio_stream = stream_index
    return audio_stream, video_stream, subtitle_stream


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

    if new_media_path.exists():
        raise RuntimeError("Save dst already exists")

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
        # h264 encode the video output
        "-vcodec",
        "libx264",
        # Preserve audio codec (but maybe it should be ac3 for ps4?)
        "-acodec",
        "copy",
        "-crf",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "28",
        new_media_path.as_posix(),
    ]
    subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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

        if adjusted_gpu_frame_allotment_count != gpu_start_frame:
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

    # Start upscaling
    with Pool(processes=len(thread_work_args)) as pool:
        pool.map(_upscale_media_on_gpu, thread_work_args)
    print("upscaling complete")

    # Build the pngs into the final media file
    _stitch_audio_from_original(media_path, png_output_dir, output_path, fps * ai_model.output_fps_increase)


if __name__ == "__main__":

    input_path = Path("//DESKTOP-0TN2T9A/g/Video/MONK/s01/disk01/title_t02.mkv")
    if not input_path.exists():
        raise RuntimeError("Failed to find input file")

    output_path = Path("g:/monk_s01e03.mkv")
    if output_path.exists():
        shutil.rmtree(output_path.as_posix(), ignore_errors=True)

    upscale_media(input_path, output_path, ai_model=AI_MODEL_DIONE_INTERLACED_ROBUST)
