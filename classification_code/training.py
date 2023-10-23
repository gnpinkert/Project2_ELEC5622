from extras import create_logger, args
import logging
import torch
from torch import nn
import torch.optim as optim
from loaders import create_data_loader, LoaderType

from network import AlexNet


def train_net(net, trainloader, valloader, logging, criterion, optimizer, scheduler, epochs=1):

    for epoch in range(epochs):  # loop over the dataset multiple times, only 1 time by default
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
            if i % 20 == 19:    # print every 2000 mini-batches
                logging.info('[%d, %5d] Training loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        running_loss = 0.0
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


    # save network
    torch.save(net, 'project2_modified.pth')
    # write finish to the file
    logging.info('Finished Training')


def main():
    network = AlexNet()
    if args.cuda:
        network = network.cuda()
    training_loader = create_data_loader(LoaderType.TRAIN)
    validation_loader = create_data_loader(LoaderType.VALIDATION)

    logger = create_logger('./')
    logger.info('using args:')
    logger.info(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.0004, momentum=0.9)  # adjust optimizer settings

    for param in network.alexnet.parameters():
        param.requires_grad = False

    train_net(net=network, trainloader=training_loader, valloader=validation_loader, criterion=criterion, optimizer=optimizer, scheduler=None, epochs=10, logging=logging)

if __name__=="__main__":
    main()
