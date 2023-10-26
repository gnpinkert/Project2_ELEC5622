from extras import create_logger, args
from network import AlexNet
from NetworkDetails import TrainingDetails
from loaders import create_data_loader, LoaderType
import torch
from pathlib import Path
from extras import get_repo_root_dir
import time



def eval_net(net, loader, final_output_path: Path):
    logger = create_logger(args.output_path, final_output_path=final_output_path)
    logger.info('using args:')
    logger.info(args)
    
    total_time = 0.0

    net = net.eval()
    if args.cuda:
        net = net.cuda()

    # use your trained network by default
    model_path = final_output_path / 'project2.pth'

    if args.cuda:
        net.load_state_dict(torch.load(model_path, map_location='cuda'))
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        start = time.process_time()
        outputs = net(images)
        end = time.process_time()
        total_time += end - start
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    inf_time = total_time / len(loader.dataset)

    # print and write to log. DO NOT CHANGE HERE.
    logger.info('=' * 55)
    logger.info('SUMMARY of Project2')
    logger.info('The number of testing image is {}'.format(total))
    logger.info('Accuracy of the network on the test images: {} %'.format(100 * round(correct / total, 4)))
    logger.info('Average Inference Time: {}'.format(inf_time))
    logger.info('=' * 55)


def main():

    training_data = TrainingDetails(batch_size=4,
                                    learning_rate=0.0004,
                                    momentum=0.9,
                                    epochs=1,
                                    output_dir=Path())
    target_dir = get_repo_root_dir() / "models" / str(training_data) / "14_30_50"

    if not target_dir.exists():
        raise FileNotFoundError(f"Can't find existing model since target dir: \"{target_dir}\" does not exist")

    test_loader = create_data_loader(LoaderType.TEST)
    network = AlexNet()
    eval_net(net=network, loader=test_loader,final_output_path=target_dir)


if __name__ == "__main__":
    main()


