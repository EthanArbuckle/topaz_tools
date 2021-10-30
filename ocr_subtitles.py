import subprocess
import tempfile
from pathlib import Path
from stream_indexer import determine_stream_indexes


def _dvdsub_to_srt(input_media: Path, output_srt: Path) -> None:
    cmds = [
        "C:/Users/user/Downloads/SubtitleEdit-3.6.2/SubtitleEdit.exe",
        "/convert",
        input_media.as_posix(),
        "subrip",
        "/ocrengine:tesseract",
        f"/outputfolder:{output_srt.as_posix()}",
    ]
    subprocess.run(cmds)


def _replace_subtitles_with_srt(input_media: Path, srt_file: Path, output_media: Path) -> None:
    # Figure out where the streams are located within the original track
    audio_stream, video_stream, _ = determine_stream_indexes(input_media)
    if not video_stream or not audio_stream:
        raise RuntimeError("Failed to find video or audio streams")

    cmds = [
        "ffmpeg",
        "-i",
        input_media.as_posix(),
        "-i",
        srt_file.as_posix(),
        "-map",
        f"0:{video_stream[-1]}",
        "-map",
        f"0:{audio_stream[-1]}",
        "-map",
        "1:0",
        "-c",
        "copy",
        output_media.as_posix(),
    ]

    subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def replace_subtitles_in_media(input_media: Path) -> None:
    if not input_media.exists():
        raise RuntimeError("Provided file does not exist")

    with tempfile.TemporaryDirectory() as tempdir:
        tmpdir_path = Path(tempdir)
        _dvdsub_to_srt(input_media, tmpdir_path)

        srt_path = list(tmpdir_path.glob("*.srt"))[0]
        if not srt_path:
            raise RuntimeError("Failed to generate SRT")

        output_media_file = input_media.parent / f"{input_media.stem}_srt.{input_media.suffix}"
        _replace_subtitles_with_srt(input_media, srt_path, output_media_file)


if __name__ == "__main__":
    media = Path("G:\monk_s01e03.mkv")
    replace_subtitles_in_media(media)
