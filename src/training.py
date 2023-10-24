from extras import create_logger, args, make_output_directory
import logging
import torch
from NetworkDetails import TrainingDetails, save_model_information
from torch import nn
import torch.optim as optim
from loaders import create_data_loader, LoaderType, BATCH_SIZE
from pathlib import Path

from network import AlexNet


def train_net(net, trainloader, valloader, logging, training_details: TrainingDetails):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(),
                          lr=training_details.learning_rate,
                          momentum=training_details.momentum)  # adjust optimizer settings
    scheduler = None
    for param in net.alexnet.parameters():
        param.requires_grad = False

    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(training_details.epochs):  # loop over the dataset multiple times, only 1 time by default
        running_loss = 0.0
        current_training_loss = 0.0
        current_training_accuracy = 0.0
        net = net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.cuda:
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            current_training_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                logging.info('[%d, %5d] Training loss: %.3f' %
                             (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            current_training_accuracy += (outputs.argmax(1) == labels).sum().item()
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        current_training_loss = current_training_loss / len(trainloader.dataset)
        training_loss.append(current_training_loss)
        current_training_accuracy = 100 * current_training_accuracy / len(trainloader.dataset)
        training_accuracy.append(current_training_accuracy)

        current_validation_loss = 0.0
        correct = 0
        net = net.eval()
        for i, data in enumerate(valloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimize
            with torch.no_grad():
                outputs = net(inputs)
            loss = criterion(outputs, labels)

            if args.cuda:
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                logging.info('[%d, %5d] Validation loss: %.3f' %
                             (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            correct += (outputs.argmax(1) == labels).sum().item()

        current_validation_loss = current_validation_loss / len(valloader.dataset)
        validation_loss.append(current_validation_loss)
        current_validation_accuracy = 100 * correct / len(valloader.dataset)
        validation_accuracy.append(current_validation_accuracy)

    # save network
    save_model_information(network=net,
                           output_directory=training_details.output_dir,
                           optimizer=optimizer,
                           criterion=criterion,
                           scheduler=scheduler,
                           validation_loss=validation_loss,
                           validation_accuracy=validation_accuracy,
                           training_loss=training_loss,
                           training_accuracy=training_accuracy)
    # write finish to the file
    logging.info('Finished Training')


def main():
    network = AlexNet()
    if args.cuda:
        network = network.cuda()
    training_loader = create_data_loader(LoaderType.TRAIN)
    validation_loader = create_data_loader(LoaderType.VALIDATION)

    training_details = TrainingDetails(batch_size=BATCH_SIZE,
                                       learning_rate=0.0004,
                                       momentum=0.9,
                                       epochs=10,
                                       output_dir=Path(""))

    output_directory = make_output_directory(training_details=training_details)

    training_details.output_dir = output_directory

    logger = create_logger(training=True, final_output_path=output_directory)
    logger.info('using args:')
    logger.info(args)

    train_net(net=network,
              trainloader=training_loader,
              valloader=validation_loader,
              training_details=training_details,
              logging=logging)


if __name__ == "__main__":
    main()