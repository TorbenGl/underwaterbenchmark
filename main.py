from lightning.pytorch import cli
import logging
import torch
import lightning

from lightning.pytorch.callbacks import ModelSummary

class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.suppress_errors = True  # type: ignore
        super().__init__(*args, **kwargs)


    def add_arguments_to_parser(self, parser):
        parser.add_argument("model.test", type=int)


def cli_main():
    LightningCLI(
        lightning.LightningModule,
        lightning.LightningDataModule,
        subclass_mode_model=False,
        subclass_mode_data=False,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "log_every_n_steps": 1,
            "enable_model_summary": False,
            "callbacks": [ModelSummary(max_depth=2)],
            "devices": "auto",
            "accumulate_grad_batches": 16,
        },
    )


if __name__ == "__main__":    
    cli_main()
    