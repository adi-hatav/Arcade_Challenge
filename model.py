import torch
import torch.nn as nn
import pytorch_lightning as pl
from arcade_dataset import load_dataset

H_in, W_in = 512, 512


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gn_groups=8):
        super(ResBlock, self).__init__()
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.block = nn.Sequential(
            nn.GroupNorm(gn_groups, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(gn_groups, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        res = self.residual(x)
        x = self.block(x)
        x = x + res
        return x


class UNetEncoder(nn.Module):
    def __init__(
        self, in_channels, out_shape=(1, H_in, W_in), gn_groups=8, n_init_features=32
    ):
        super(UNetEncoder, self).__init__()
        H_in, W_in = out_shape[1], out_shape[2]
        self.initial_conv = nn.Conv2d(
            in_channels, n_init_features, kernel_size=3, stride=1, padding=1
        )
        self.blocks = nn.ModuleList(
            [
                ResBlock(n_init_features * 1, n_init_features * 1, gn_groups),
                nn.Sequential(
                    ResBlock(n_init_features * 1, n_init_features * 2, gn_groups),
                    ResBlock(n_init_features * 2, n_init_features * 2, gn_groups),
                ),
                nn.Sequential(
                    ResBlock(n_init_features * 2, n_init_features * 4, gn_groups),
                    ResBlock(n_init_features * 4, n_init_features * 4, gn_groups),
                ),
                nn.Sequential(
                    ResBlock(n_init_features * 4, n_init_features * 8, gn_groups),
                    ResBlock(n_init_features * 8, n_init_features * 8, gn_groups),
                    ResBlock(n_init_features * 8, n_init_features * 8, gn_groups),
                ),
                nn.Sequential(
                    ResBlock(n_init_features * 8, n_init_features * 8, gn_groups),
                ),
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.AdaptiveMaxPool2d((H_in // 2, W_in // 2)),
                nn.AdaptiveMaxPool2d((H_in // 4, W_in // 4)),
                nn.AdaptiveMaxPool2d((H_in // 8, W_in // 8)),
                nn.AdaptiveMaxPool2d((H_in // 16, W_in // 16)),
            ]
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = nn.Dropout2d(0.2)(x)
        x = self.blocks[0](x)
        skips = []
        for block, downsample in zip(self.blocks[1:], self.downsamples):
            skips.append(x)
            x = block(x)
            x = downsample(x)
        return x, skips


class UNetDecoder(nn.Module):
    def __init__(self, out_shape, gn_groups=8, n_init_features=32):
        super(UNetDecoder, self).__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(n_init_features * 8, n_init_features * 8, gn_groups),
                ResBlock(n_init_features * 4, n_init_features * 4, gn_groups),
                ResBlock(n_init_features * 2, n_init_features * 2, gn_groups),
                ResBlock(n_init_features * 1, n_init_features * 1, gn_groups),
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Upsample(scale_factor=2, mode="bilinear"),
            ]
        )
        self.downsize_features = nn.ModuleList(
            [
                nn.Conv2d(
                    n_init_features * 8, n_init_features * 8, kernel_size=1, padding=0
                ),
                nn.Conv2d(
                    n_init_features * 8, n_init_features * 4, kernel_size=1, padding=0
                ),
                nn.Conv2d(
                    n_init_features * 4, n_init_features * 2, kernel_size=1, padding=0
                ),
                nn.Conv2d(
                    n_init_features * 2, n_init_features * 1, kernel_size=1, padding=0
                ),
            ]
        )
        self.final_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    n_init_features * 1,
                    n_init_features * 1,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),
                nn.Conv2d(
                    n_init_features * 1, out_shape[0], kernel_size=1, padding="same"
                ),
            ]
        )

    def forward(self, x, skips):
        for block, upsample, downsample_channels, skip in zip(
            self.blocks, self.upsamples, self.downsize_features, reversed(skips)
        ):
            x = downsample_channels(x)
            x = upsample(x)
            x = x + skip
            x = block(x)
        for final_conv in self.final_convs:
            x = final_conv(x)
        return x


class VAEDecoder(nn.Module):
    def __init__(self, gn_groups=8, n_init_features=32):
        super(VAEDecoder, self).__init__()
        self.initial_layers = nn.Sequential(
            nn.GroupNorm(gn_groups, n_init_features * 8),
            nn.ReLU(),
            nn.Conv2d(
                n_init_features * 8,
                n_init_features // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Flatten(),
            nn.Linear(n_init_features // 2 * H_in // 32 * W_in // 32, 256),
        )
        self.mu = nn.Linear(256, 128)
        self.logvar = nn.Linear(256, 128)
        self.sample = lambda mu, logvar: mu + torch.randn_like(logvar) * torch.exp(
            0.5 * logvar
        )
        self.upsample = nn.Sequential(
            nn.Linear(128, n_init_features // 4 * H_in // 16 * W_in // 16),
            nn.ReLU(),
            nn.Unflatten(1, (n_init_features // 4, H_in // 16, W_in // 16)),
            nn.Conv2d(
                n_init_features // 4,
                n_init_features * 8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                n_init_features * 8,
                n_init_features * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ResBlock(n_init_features * 4, n_init_features * 4, gn_groups),
            nn.Conv2d(
                n_init_features * 4,
                n_init_features * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ResBlock(n_init_features * 2, n_init_features * 2, gn_groups),
            nn.Conv2d(
                n_init_features * 2,
                n_init_features * 1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ResBlock(n_init_features * 1, n_init_features * 1, gn_groups),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                n_init_features * 1,
                n_init_features * 1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                n_init_features * 1, 1, kernel_size=1
            ),
        )

    def forward(self, x):
        x = self.initial_layers(x)
        mu, logvar = self.mu(x), self.logvar(x)
        x = self.sample(mu, logvar)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x, [mu, logvar]


class LabelClassifier(nn.Module):
    def __init__(self, in_channels, n_classes, gn_groups=8, n_init_features=32, drop=0.4):
        super(LabelClassifier, self).__init__()
        self.initial_layers = nn.Sequential(
            nn.GroupNorm(gn_groups, n_init_features * 8),
            nn.ReLU(),
            nn.Conv2d(
                n_init_features * 8,
                n_init_features // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Flatten(),
            nn.Linear(n_init_features // 2 * H_in // 32 * W_in // 32, 256),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class UNet(nn.Module):
    def __init__(
        self, in_channels, out_shape=(25, 512, 512), gn_groups=8, n_init_features=32
    ):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, out_shape, gn_groups, n_init_features)
        self.decoder = UNetDecoder(out_shape, gn_groups, n_init_features)
        self.vae_decoder = VAEDecoder(gn_groups, n_init_features)
        self.label_classifier = LabelClassifier(in_channels, out_shape[0], gn_groups, n_init_features)

    def forward(self, x):
        encoder_output, skips = self.encoder(x)
        vae_output, [mu, logvar] = self.vae_decoder(encoder_output)
        decoder_output = self.decoder(encoder_output, skips)
        decoder_output = torch.sigmoid(decoder_output)
        label_output = self.label_classifier(encoder_output)
        reconstruction_loss = nn.MSELoss()(
            vae_output, x[:, 0, :, :].unsqueeze(1)
        )
        return decoder_output, vae_output, [mu, logvar], label_output, reconstruction_loss


class VesselSegmentationModel(pl.LightningModule):
    def __init__(
        self, in_channels=3, out_shape=(25, 512, 512), gn_groups=8, n_init_features=32
    ):
        super(VesselSegmentationModel, self).__init__()
        self.model = UNet(in_channels, out_shape, gn_groups, n_init_features)
        self.multi_class = out_shape[0] > 1
        self.n_classes = out_shape[0]

    def _dice_loss(self, y_gt, y_pred, multi_class=False, eps=1e-6):
        if not multi_class:
            y_gt = y_gt.view(-1)
            y_pred = y_pred.view(-1)
            intersection = torch.sum(y_gt * y_pred)
            dice_score = (eps + 2 * intersection) / (torch.sum(y_gt) + torch.sum(y_pred) + eps)
            return 1 - dice_score
        else:
            y_gt = y_gt.view(-1, self.n_classes)
            y_pred = y_pred.view(-1, self.n_classes)
            intersection = torch.sum(y_gt * y_pred, dim=0)
            dice_score = (1 + 2 * intersection) / (
                torch.sum(y_gt, dim=0) + torch.sum(y_pred, dim=0) + 1
            )
            return 1 - dice_score.mean()

    def _tversky_loss(self, y_gt, y_pred, alpha=0.4, beta=0.6, eps=1e-6, multi_class=False):
        if not multi_class:
            y_gt = y_gt.view(-1)
            y_pred = y_pred.view(-1)
            tp = torch.sum(y_gt * y_pred)
            fp = torch.sum((1 - y_gt) * y_pred)
            fn = torch.sum(y_gt * (1 - y_pred))
            tversky_score = (eps + tp) / (eps + tp + alpha * fp + beta * fn)
        else:
            y_gt = y_gt.view(-1, self.n_classes)
            y_pred = y_pred.view(-1, self.n_classes)
            tp = torch.sum(y_gt * y_pred, dim=0)
            fp = torch.sum((1 - y_gt) * y_pred, dim=0)
            fn = torch.sum(y_gt * (1 - y_pred), dim=0)
            tversky_score = (eps + tp) / (eps + tp + alpha * fp + beta * fn)
        return 1 - tversky_score.mean()

    def _focal_loss(self, y_gt, y_pred, alpha=0.25, gamma=2):
        y_gt = y_gt.view(-1)
        y_pred = y_pred.view(-1)

        BCE_loss = torch.nn.functional.binary_cross_entropy(y_pred, y_gt, reduction="mean")
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss
        return F_loss.mean()

    def _bce_loss(self, y_gt, y_pred, multi_class=False):
        return torch.functional.F.binary_cross_entropy(y_pred, y_gt, reduction="mean")
        
    def _label_cross_entropy_loss(self, y_gt, y_pred):
        return torch.nn.functional.binary_cross_entropy(y_pred, y_gt)

    def forward(self, x):
        return self.model(x)

    def _loss(
        self,
        x,
        y,
        labels_gt,
        lambda_kl=0.1,
        lambda_l2=0.1,
        lambda_bce=0.5,
        lambda_dice=1.,
        lambda_tversky=0.,
        lambda_label=0.5,
        lambda_focal=0.,
    ):
        y_hat, _, [mu, logvar], labels, reconstruction_loss = self.model(x)
        kl_loss = -(1 / x.numel()) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        bce_loss = self._bce_loss(y, y_hat, multi_class=self.multi_class)
        dice_loss = self._dice_loss(y, y_hat, multi_class=self.multi_class)
        # tversky_loss = self._tversky_loss(y, y_hat, multi_class=self.multi_class)
        # focal_loss = self._focal_loss(y, y_hat)
        tversky_loss = 0
        focal_loss = 0
        classification_loss = self._label_cross_entropy_loss(labels_gt, labels)
        loss = (
            + lambda_kl * kl_loss
            + lambda_l2 * reconstruction_loss
            + lambda_bce * bce_loss
            + lambda_dice * dice_loss
            + lambda_tversky * tversky_loss
            + lambda_label * classification_loss
            + lambda_focal * focal_loss
        )
        return loss, kl_loss, reconstruction_loss, bce_loss, dice_loss, tversky_loss, focal_loss, classification_loss

    def training_step(self, batch, batch_idx):
        reconstruction_lambda = 0.1 if batch_idx <= 1000 * 5 / 8 else 0.01
        loss, kl_loss, reconstruction_loss, bce_loss, dice_loss, tversky_loss, focal_loss, classification_loss = (
            self._loss(batch["transformed_image"], batch["separate_masks"], batch["labels"], lambda_l2=reconstruction_lambda)
        )
        self.log("train_loss", loss)
        self.log("kl_loss", kl_loss)
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("bce_loss", bce_loss)
        self.log("dice_loss", dice_loss)
        self.log("tversky_loss", tversky_loss)
        self.log("focal_loss", focal_loss)
        self.log("classification_loss", classification_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def train_dataloader(self, batch_size=8):
        return torch.utils.data.DataLoader(
            load_dataset("train"), batch_size=batch_size, num_workers=7, shuffle=False
        )

    def val_dataloader(self, batch_size=8):
        return torch.utils.data.DataLoader(
            load_dataset("val"), batch_size=batch_size, num_workers=7, shuffle=False
        )

    def validation_step(self, batch, batch_idx):
        loss = self._dice_loss(
            batch["separate_masks"], self.model(batch["transformed_image"])[0]
        )
        self.log("val_loss", loss)
        return loss


if __name__ == "__main__":
    # Clear the cuda cache
    torch.cuda.empty_cache()

    # Create the model
    model = VesselSegmentationModel(in_channels=3)
    
    # Train the model for 10 epochs on the entire data
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath="./models/first_run",
            save_top_k=3,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        # pl.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     patience=5,
        #     mode="min",
        # ),
    ]
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',
        callbacks=callbacks,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
        # fast_dev_run=True
    )
    trainer.fit(model)