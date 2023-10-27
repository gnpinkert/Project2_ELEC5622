import torchvision.transforms as transforms
from pathlib import Path
from bidict import bidict
from typing import List, Tuple, Dict, Any
from enum import Enum
from PIL import Image
import csv
import torchvision.transforms.v2
import torch
from torch.utils.data import Dataset, DataLoader
from extras import get_repo_root_dir

BATCH_SIZE: int = 4
NUMBER_OF_WORKERS: int = 2

LABEL_MAP = bidict({"Homogeneous": 0,
                    "Speckled": 1,
                    "Nucleolar": 2,
                    "Centromere": 3,
                    "NuMem": 4,
                    "Golgi": 5})

class ProjectData(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data: List[Path] = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample: Path = self.data[index]
        filename = str(sample.stem)
        image = Image.open(sample).convert("L")
        label = self.labels[str(int(filename))]

        if self.transform:
            image = self.transform(image)
        image = image.expand(3, -1, -1)
        return image, LABEL_MAP[label]


class LoaderType(Enum):
    TEST = 1
    TRAIN = 2
    VALIDATION = 3


_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

_train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(size=224, scale=(0.8, 0.8)),
    # transforms.RandomRotation(degrees=180,expand=False),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])



def _load_label_csv():
    repo_root_dir = get_repo_root_dir()
    csv_file_name = "gt_training.csv"

    label_csv_path: Path = repo_root_dir / "training_data" / csv_file_name
    if not label_csv_path.exists():
        raise RuntimeError(f"File {csv_file_name} not found in {label_csv_path.stem}")
    labels_dict: Dict[str, str] = {}

    with open(label_csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            labels_dict[line[0]] = line[1]
    return labels_dict


def _get_image_filepaths(data_directory: Path) -> List[Path]:
    repo_root_dir = get_repo_root_dir()
    data_dir = repo_root_dir / data_directory

    if not data_dir.exists():
        raise RuntimeError(f"Invalid data directory: {data_dir}")

    file_paths: List[Path] = []
    for file in data_dir.rglob('*.*'):
        file_paths.append(file)
    return file_paths


def _get_data_dir_and_transform(loader_enum: LoaderType) -> Tuple[Path, torchvision.transforms.Compose]:
    data_dir = Path("training_data")
    match loader_enum:
        case loader_enum.TEST:
            return data_dir / "test", _test_transform
        case loader_enum.TRAIN:
            return data_dir / "training", _train_transform
        case loader_enum.VALIDATION:
            return data_dir / "validation", _test_transform
        case _:
            raise RuntimeError(f"Invalid enum type: {loader_enum}")


def create_data_set(loader_enum: LoaderType, custom_transform: transforms.Compose = None) -> ProjectData:
    data_dir, transform = _get_data_dir_and_transform(loader_enum)
    transform = custom_transform if custom_transform is not None else transform
    file_paths = _get_image_filepaths(data_dir)
    return ProjectData(data=file_paths,
                       labels=_load_label_csv(),
                       transform=transform)


def create_data_loader(loader_enum: LoaderType, transform: transforms.Compose = None) -> DataLoader[Any]:
    data_set = create_data_set(loader_enum=loader_enum, custom_transform=transform)
    return torch.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMBER_OF_WORKERS)


def load_all_datasets() -> List[Tuple[DataLoader, str]]:
    data_sets: List[Tuple[DataLoader, str]] = []
    for data_set_type in LoaderType:
        data_sets.append((create_data_loader(data_set_type), data_set_type.name))
    return data_sets


def main():
    _load_label_csv()
    data_set_tuples = load_all_datasets()

    for data_set, data_type in data_set_tuples:
        print(f"There are {len(data_set)} with batch size: {BATCH_SIZE} in data set: {data_type}")


if __name__ == "__main__":
    main()
