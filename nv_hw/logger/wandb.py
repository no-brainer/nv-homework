from datetime import datetime

import PIL

from .util import plot_spectrogram_to_buf


class WanDBWriter:
    def __init__(self, config, logger):
        self.writer = None
        self.selected_module = ""

        try:
            import wandb
            wandb.login()

            if config["trainer"].get("wandb_project") is None:
                raise ValueError("please specify project name for wandb")

            wandb.init(
                project=config["trainer"].get("wandb_project"),
                config=config.config
            )
            self.wandb = wandb

        except ImportError:
            logger.warning("For use wandb install it via \n\t pip install wandb")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def scalar_name(self, scalar_name):
        return f"{scalar_name}_{self.mode}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self.scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        buf = plot_spectrogram_to_buf(image.detach().cpu().T)
        img = PIL.Image.open(buf)
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Image(img)
        }, step=self.step)
        img.close()
        buf.close()

    def add_audio(self, scalar_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)
