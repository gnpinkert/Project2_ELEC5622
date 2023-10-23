from extras import create_logger, args
import logging
from network import AlexNet
from loaders import create_data_loader, LoaderType
import torch

def eval_net(net, loader, logger):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    # use your trained network by default
    model_name = './project2_modified.pth'

    if args.cuda:
        net.load_state_dict(torch.load(model_name, map_location='cuda'))
    else:
        net.load_state_dict(torch.load(model_name, map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log. DO NOT CHANGE HERE.
    logging.info('=' * 55)
    logging.info('SUMMARY of Project2')
    logger.info('The number of testing image is {}'.format(total))
    logging.info('Accuracy of the network on the test images: {} %'.format(100 * round(correct / total, 4)))
    logging.info('=' * 55)
def main():
    logger = create_logger(args.output_path)
    logger.info('using args:')
    logger.info(args)

    test_loader = create_data_loader(LoaderType.TEST)
    network = AlexNet()

    eval_net(net=network, loader=test_loader, logger=logger)

if __name__ == "__main__":
    main()


