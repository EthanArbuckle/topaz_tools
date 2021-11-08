from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AiModel:
    short_name: str
    output_fps_increase: int


@dataclass
class Gpu:
    name: str
    cuda_index: int
    remote_host: Optional[str] = None

    def __str__(self) -> str:
        origin = "remote" if self.remote_host else "local"
        return f"[{origin} {self.name}]"


@dataclass
class TopazUpscaleJob:
    source_media_path: Path
    output_media_path: Path
    job_start_time: float


@dataclass
class FrameUpscalingJob(TopazUpscaleJob):
    png_output_path: Path
    start_frame: int
    end_frame: Optional[int]
    ai_model: AiModel
    output_scale: float
    total_chunk_count: int
    output_fps: float


@dataclass
class MediaUpscaleJob(TopazUpscaleJob):
    png_output_path_root: Path
    ai_model: AiModel
    output_scale: float
    chunk_size: int = 250


@dataclass
class CompletedChunk(TopazUpscaleJob):
    chunk_png_output_path: Path
    total_chunk_count: int
    output_fps: float


@dataclass
class PngEncodingJob(TopazUpscaleJob):
    png_output_path_root: Path
    output_fps: float
