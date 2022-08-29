import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from project.datamodule import LicensePlateRecognitionDataModule
from project.lit_recognition import LicensePlateRecognitionModule


if __name__ == '__main__':
    model_checkpoint = ModelCheckpoint(
        dirpath='model_weights/',
        filename='plates_recognition-{epoch:02d}.ckpt',
        monitor='valid/acc',
        save_top_k=3,
        verbose=True,
        mode='min',
    )

    logger = WandbLogger(name='license_plate_recognition', project='image_processing_cnn')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=20,
        callbacks=[model_checkpoint],
        logger=logger,
        log_every_n_steps=50
    )

    data = LicensePlateRecognitionDataModule(
        root_dir='data/license_plates_recognition_train',
        metadata_file='data/license_plates_recognition_train.csv'
    )

    model = LicensePlateRecognitionModule()

    trainer.fit(model, data)
    trainer.test(model, data)
