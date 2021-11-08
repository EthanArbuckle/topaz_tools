import subprocess
from pathlib import Path
from typing import Optional, Tuple


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
