from extras import create_logger, args, make_output_directory
import torch
import copy
from NetworkDetails import TrainingDetails, save_model_information
from torch import nn
import torch.optim as optim
from loaders import create_data_loader, LoaderType, BATCH_SIZE
from pathlib import Path
import time

from network import AlexNet

def train_net(net, trainloader, valloader, logging, training_details:TrainingDetails, print_every_samples = 50, patience = 3):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(),
                          lr=training_details.learning_rate,
                          momentum=training_details.momentum)  # adjust optimizer settings
    scheduler = None
    for param in net.alexnet.parameters():
         param.requires_grad = False
    for param in net.alexnet.classifier.parameters():
        param.requires_grad = True

    validation_loss_list = []
    training_loss_list = []
    validation_accuracy_list = []
    training_accuracy_list = []

    best_state_dictionary = None
    best_validation_accuracy = 0.0
    inertia = 0
    start = time.process_time()
    for epoch in range(training_details.epochs):  # loop over the dataset multiple times, only 1 time by default

        training_loss = 0.0
        training_accuracy = 0.0
        running_loss = 0.0
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
            training_loss += loss.item()
            if i % print_every_samples == print_every_samples - 1:    # print every 2000 mini-batches
                logging.info('[%d, %5d] Training loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every_samples))
                running_loss = 0.0

            training_accuracy += (outputs.argmax(1) == labels).sum().item()

        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        training_loss = training_loss / len(trainloader.dataset)
        training_loss_list.append(training_loss)
        training_accuracy = 100 * training_accuracy / len(trainloader.dataset)
        training_accuracy_list.append(training_accuracy)

        running_loss = 0.0
        val_loss = 0.0
        correct = 0
        net = net.eval()
        for i, data in enumerate(valloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if args.cuda:
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            val_loss += loss.item()
            if i % print_every_samples == print_every_samples - 1:  # print every 2000 mini-batches
                logging.info('[%d, %5d] Validation loss: %.3f' %
                             (epoch + 1, i + 1, running_loss / print_every_samples))
                running_loss = 0.0
            correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss / len(valloader.dataset)
        validation_loss_list.append(val_loss)
        val_accuracy = 100 * correct / len(valloader.dataset)
        validation_accuracy_list.append(val_accuracy)

        if val_accuracy > best_validation_accuracy:
            best_state_dictionary = copy.deepcopy(net.state_dict())
            inertia = 0
        else:
            inertia += 1
            if inertia == patience:
                if best_state_dictionary is None:
                    raise Exception("State dictionary should have been updated at least once")
                break
        print(f"Validation accuracy: {val_accuracy}")

    end = time.process_time()
    training_time = end - start
    logging.info('Training Time: {}'.format(training_time))
    
    
    # Apply best weights
    net.load_state_dict(best_state_dictionary)

    # save network
    save_model_information(network=net,
                           output_directory=training_details.output_dir,
                           optimizer=optimizer,
                           criterion=criterion,
                           scheduler=scheduler,
                           validation_loss=validation_loss_list,
                           validation_accuracy=validation_accuracy_list,
                           training_loss=training_loss,
                           training_accuracy=training_accuracy)

    logging.info('Finished Training')
    
    return

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
              logging=logger)


if __name__ == "__main__":
    main()
