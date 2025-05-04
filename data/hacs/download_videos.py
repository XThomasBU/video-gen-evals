import os
import subprocess
import shutil
import imageio
import numpy as np
import json


class YouTubeVideoProcessor:
    def __init__(self, youtube_id, output_dir, start_frame=None, end_frame=None):
        self.youtube_id = youtube_id
        self.output_dir = output_dir
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.full_frames_dir = os.path.join(output_dir, "full_frames")
        self.selected_frames_dir = os.path.join(output_dir, "selected_frames")
        self.video_path = os.path.join(output_dir, f"{youtube_id}.mp4")
        self.full_video_path = os.path.join(output_dir, f"{youtube_id}_full.mp4")
        self.selected_video_path = os.path.join(
            output_dir, f"{youtube_id}_selected.mp4"
        )
        self.metadata_path = os.path.join(output_dir, "metadata.json")

        self._prepare_folders()

    def _prepare_folders(self):
        self._clean_folder(self.output_dir)
        self._clean_folder(self.full_frames_dir)
        self._clean_folder(self.selected_frames_dir)

    @staticmethod
    def _clean_folder(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def download_video(self):
        print("Downloading video...")
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                "-o",
                self.video_path,
                f"https://www.youtube.com/watch?v={self.youtube_id}",
            ],
            check=True,
        )
        print("Download completed.")

    def extract_frames(self):
        print("Extracting frames with ffmpeg...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                self.video_path,
                os.path.join(self.full_frames_dir, "frame_%05d.png"),
            ],
            check=True,
        )
        print("Frame extraction completed.")

    @staticmethod
    def _get_fps(video_path):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        frame_rate_raw = result.stdout.strip()
        num, den = map(int, frame_rate_raw.split("/"))
        return num / den

    @staticmethod
    def _get_duration(video_path):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    @staticmethod
    def _load_frames_from_folder(folder, start_idx=None, end_idx=None):
        frame_files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        if start_idx is not None and end_idx is not None:
            frame_files = frame_files[start_idx - 1 : end_idx]  # 1-indexed
        frames = []
        for f in frame_files:
            img = imageio.imread(os.path.join(folder, f))
            img = np.clip(img, 0, 255).astype(np.uint8)
            frames.append(img)
        frames = np.stack(frames)
        return frames

    @staticmethod
    def _save_video_imageio(frames, output_path, fps):
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    def _save_metadata(self, fps, duration, total_frames, selected_frames=None):
        metadata = {
            "youtube_id": self.youtube_id,
            "fps": fps,
            "duration_seconds": duration,
            "total_frames": total_frames,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "selected_frame_count": selected_frames if selected_frames else 0,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {self.metadata_path}")

    def process(self):
        self.download_video()
        self.extract_frames()

        fps = self._get_fps(self.video_path)
        duration = self._get_duration(self.video_path)
        total_frames = int(np.ceil(duration * fps))

        print(
            f"FPS: {fps:.3f}, Duration: {duration:.2f}s, Expected total frames: {total_frames}"
        )

        print("Loading all frames...")
        all_frames = self._load_frames_from_folder(self.full_frames_dir)

        print(f"Saving full video with {len(all_frames)} frames...")
        self._save_video_imageio(all_frames, self.full_video_path, fps=fps)

        selected_frame_count = 0

        if self.start_frame and self.end_frame:
            print(f"Selecting frames {self.start_frame} to {self.end_frame}...")
            selected_frames = self._load_frames_from_folder(
                self.full_frames_dir, self.start_frame, self.end_frame
            )
            selected_frame_count = selected_frames.shape[0]

            print(f"Saving selected video with {selected_frame_count} frames...")
            self._save_video_imageio(selected_frames, self.selected_video_path, fps=fps)

            print("Saving selected frames as images...")
            self._clean_folder(self.selected_frames_dir)
            for idx, frame in enumerate(selected_frames, start=1):
                save_path = os.path.join(
                    self.selected_frames_dir, f"frame_{idx:05d}.png"
                )
                imageio.imwrite(save_path, frame)

        self._save_metadata(fps, duration, total_frames, selected_frame_count)
        print("Processing complete.")


if __name__ == "__main__":

    filtered_hacs_id_json = "data/hacs/files/filtered_hacs.json"

    with open(filtered_hacs_id_json, "r") as f:
        filtered_hacs = json.load(f)

    for hacs in filtered_hacs:
        youtube_id = hacs["youtube_id"]
        start_sec = float(hacs["start"])
        end_sec = float(hacs["end"])

        output_dir = os.path.join("saved_data", "hacs", youtube_id)
        print(f"Downloading video ID: {youtube_id} | Time: {start_sec}-{end_sec} sec")

        processor = YouTubeVideoProcessor(youtube_id=youtube_id, output_dir=output_dir)
        processor.download_video()

        fps = processor._get_fps(processor.video_path)
        duration = processor._get_duration(processor.video_path)

        start_frame = int(np.floor(start_sec * fps)) + 1
        end_frame = int(np.ceil(end_sec * fps))

        print(f"Computed frame range: {start_frame} to {end_frame} at {fps:.2f} FPS")

        processor = YouTubeVideoProcessor(
            youtube_id=youtube_id,
            output_dir=output_dir,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        processor.process()

    print("All videos downloaded and processed.")
