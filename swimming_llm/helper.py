import numpy as np
import mujoco
import cv2

class SimulationRecorder:
    def __init__(self, model, data, width=1280, height=720): # Set desired resolution here
        self.model = model
        self.data = data
        # Initialize renderer with specific dimensions
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.frames = []

    def record_frame(self, action_text=""):
        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # Add text overlay if action_text is provided
        if action_text:
            action_text=f"Action: {action_text}"
            self._add_text_overlay(frame_bgr, action_text)

        self.frames.append(frame_bgr)

    def _add_text_overlay(self, image, text):
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (5, 50)  # (x, y) coordinates from top-left
        font_scale = image.shape[0] / 2000.0
        color = (255, 255, 255)  # White in BGR
        cv2.putText(image, text, position, font, font_scale, (0, 0, 0))

    def save_video(self, filename, fps=30):
        if not self.frames: 
            print("No frames to save.")
            return
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        for frame in self.frames: 
            video.write(frame)
        video.release()



def format_for_print(value):
    """
    Produce shorter readable output for logs.
    """
    if isinstance(value, dict):
        return {
            key: format_for_print(item)
            for key, item in value.items()
        }

    if isinstance(value, np.ndarray):
        return format_for_print(value.tolist())

    if isinstance(value, (list, tuple)):
        return [format_for_print(item)for item in value]

    if isinstance(value,(float, np.floating),):
        return round(float(value), 3)

    return value
