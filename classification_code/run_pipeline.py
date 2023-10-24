from extras import create_logger, make_output_directory, args, register_sig_abrt, unregister_sig_abrt
from loaders import load_all_datasets
from network import AlexNet
from test import eval_net
from loaders import BATCH_SIZE
import ctypes
from NetworkDetails import TrainingDetails
import logging
from training import train_net
import shutil
from pathlib import Path



def main():

    (test_loader, _), (train_loader, _), (validation_loader, _) = load_all_datasets()

    training_details = TrainingDetails(batch_size=BATCH_SIZE,
                                       learning_rate=0.0004,
                                       momentum=0.9,
                                       epochs=10,
                                       output_dir=Path(""))
    network = AlexNet()
    output_directory = make_output_directory(training_details=training_details)
    training_details.output_dir = output_directory
    try:
        logger = create_logger(training=True, final_output_path=output_directory)
        logger.info('using args:')
        logger.info(args)
        register_sig_abrt(directory_to_clean=output_directory)
        train_net(net=network,
                  trainloader=train_loader,
                  valloader=validation_loader,
                  training_details=training_details,
                  logging=logging)
        unregister_sig_abrt()
    except Exception as e:
        print(f"Caught exception {e} during training, cleaning directory {output_directory}")
        shutil.rmtree(output_directory)

    eval_net(net=network, loader=test_loader, final_output_path=output_directory)


if __name__ == "__main__":
    main()