import itertools

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from nv_hw.base import BaseTrainer
from nv_hw.losses import MelLoss, FeatureLoss, GeneratorLoss, DiscriminatorLoss
from nv_hw.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            gen_model,
            gen_opt,
            gen_scheduler,
            disc_mpd,
            disc_msd,
            disc_opt,
            disc_scheduler,
            config,
            device,
            featurizer,
            data_loader,
            data_loader_full,
            valid_data_loader=None,
            valid_data_loader_full=None,
            len_epoch=None,
            skip_oom=True,
            sr=22050
    ):
        super().__init__(gen_model, disc_mpd, disc_msd, gen_opt, disc_opt, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.log_step = 50
        self.sr = sr

        self.data_loader_full = data_loader_full
        self.valid_data_loader_full = valid_data_loader_full

        self.losses = {
            "mel_loss": MelLoss(),
            "feat_loss": FeatureLoss(),
            "gen_loss": GeneratorLoss(),
            "disc_loss": DiscriminatorLoss(),
        }

        self.featurizer = featurizer

        self.train_metrics = MetricTracker(
            "loss_g", "loss_d", "grad_norm_g", "grad_norm_mpd", "grad_norm_msd", writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss_g", "loss_d", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["waveforms_real"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, model):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.gen_model.train()
        self.disc_msd.train()
        self.disc_mpd.train()

        self.train_metrics.reset()

        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            if batch_idx >= self.len_epoch:
                break

            self.gen_opt.zero_grad()
            self.disc_opt.zero_grad()

            try:
                batch = self.process_batch(
                    batch_idx,
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in itertools.chain(
                            self.gen_model.parameters(),
                            self.disc_mpd.parameters(),
                            self.disc_msd.parameters()
                    ):
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if (batch_idx + 1) % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                loss_key = "loss_g" if batch.get("loss_g") is not None else "loss_d"
                self.logger.debug(
                    "Train Epoch: {} {} {}: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss_key, batch[loss_key]
                    )
                )
                self.writer.add_scalar(
                    "lr_d", self.disc_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "lr_g", self.gen_scheduler.get_last_lr()[0]
                )
                self._log_media(**batch)
                self._log_scalars(self.train_metrics)
                self.train_metrics.reset()

        log = self.train_metrics.result()

        self.log_inference(self.data_loader_full)
        self.disc_scheduler.step()
        self.gen_scheduler.step()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(self, batch_idx, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        batch["melspecs_real"] = self.featurizer(batch["waveforms_real"])

        batch["waveforms_fake"] = self.gen_model(batch["melspecs_real"])
        batch["melspecs_fake"] = self.featurizer(batch["waveforms_fake"])

        if batch_idx % 2:
            # discriminator step
            outs_real, outs_fake, *_ = self.disc_mpd(batch["waveforms_real"], batch["waveforms_fake"].detach())
            loss_mpd, *_ = self.losses["disc_loss"](outs_real, outs_fake)

            outs_real, outs_fake, *_ = self.disc_msd(batch["waveforms_real"], batch["waveforms_fake"].detach())
            loss_msd, *_ = self.losses["disc_loss"](outs_real, outs_fake)

            loss_d = loss_msd + loss_mpd
            if is_train:
                loss_d.backward()
                self.disc_opt.step()
                metrics.update("grad_norm_mpd", self.get_grad_norm(self.disc_mpd))
                metrics.update("grad_norm_msd", self.get_grad_norm(self.disc_msd))

            metrics.update("loss_d", loss_d.item())
            batch["loss_d"] = loss_d.item()

        else:
            # generator step
            loss_g = 45 * self.losses["mel_loss"](
                batch["melspecs_real"], batch["melspecs_fake"]
            )

            _, outs_p_fake, fmaps_p_real, fmaps_p_fake = self.disc_mpd(
                batch["waveforms_real"], batch["waveforms_fake"]
            )
            _, outs_s_fake, fmaps_s_real, fmaps_s_fake = self.disc_msd(
                batch["waveforms_real"], batch["waveforms_fake"]
            )

            loss_g += (
                self.losses["feat_loss"](fmaps_p_real, fmaps_p_fake) +
                self.losses["feat_loss"](fmaps_s_real, fmaps_s_real)
            )

            loss_gen_p, _ = self.losses["gen_loss"](outs_p_fake)
            loss_gen_s, _ = self.losses["gen_loss"](outs_s_fake)

            loss_g += loss_gen_p + loss_gen_s

            if is_train:
                loss_g.backward()
                self.gen_opt.step()
                metrics.update("grad_norm_g", self.get_grad_norm(self.gen_model))

            metrics.update("loss_g", loss_g.item())
            batch["loss_g"] = loss_g.item()

        return batch

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.gen_model.eval()
        self.disc_msd.eval()
        self.disc_mpd.eval()

        self.valid_metrics.reset()

        for batch_idx, batch in tqdm(
                enumerate(self.valid_data_loader),
                desc="validation",
                total=len(self.valid_data_loader),
        ):
            batch = self.process_batch(
                batch_idx,
                batch,
                is_train=False,
                metrics=self.valid_metrics,
            )

        self.writer.set_step(epoch * self.len_epoch, "valid")
        self._log_scalars(self.valid_metrics)
        self._log_media(**batch)

        self.log_inference(self.valid_data_loader_full)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def log_inference(self, dataloader):
        self.gen_model.eval()

        batch = self.move_batch_to_device(next(iter(dataloader)), self.device)
        batch["melspecs_real"] = self.featurizer(batch["waveforms_real"])
        batch["waveforms_fake"] = self.gen_model(batch["melspecs_real"])

        self._log_media(**batch, label="full")

    def _log_media(
            self,
            melspecs_real,
            waveforms_real,
            waveforms_fake,
            transcripts,
            label="",
            *args,
            **kwargs
    ):
        self.writer.add_audio(f"real_audio_{label}", waveforms_real[0].detach().cpu(), self.sr)
        self.writer.add_audio(f"fake_audio_{label}", waveforms_fake[0].detach().cpu(), self.sr)
        self.writer.add_text(f"transcript_{label}", transcripts[0])

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
