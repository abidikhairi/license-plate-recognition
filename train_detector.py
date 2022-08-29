import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from project.datamodule import LicensePlateDetectionDataModule
from project.lit_detector import LicensePlateDetectorModule


if __name__ == '__main__':
    model_checkpoint = ModelCheckpoint(
        dirpath='model_weights/',
        filename='plates_detector-{epoch:02d}.ckpt',
        monitor='val/loss',
        save_top_k=3,
        verbose=True,
        mode='min',
    )
    
    logger = WandbLogger(name='license_plate_detector', project='image_processing_cnn')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=30,
        callbacks=[model_checkpoint],
        logger=logger,
        log_every_n_steps=5
    )
    data = LicensePlateDetectionDataModule(
        root_dir='data/license_plates_detection_train',
        metadata_file='data/license_plates_detection_train.csv',
    )
    model = LicensePlateDetectorModule()

    trainer.fit(model, data)
    trainer.test(model, data)
