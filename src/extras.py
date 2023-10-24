from datetime import datetime
import argparse
import logging
import os.path
import shutil
from pathlib import Path
from NetworkDetails import TrainingDetails
import git
import signal
import sys

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
parser.add_argument('--output_path', default='./', type=str,
                    help='The path that stores the log files.')
args = parser.parse_args()


def get_repo_root_dir() -> Path:
    git_repo = git.Repo("./", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)


def create_logger(training: bool, final_output_path: Path):
    log_file = f"{'training' if training else 'test'}.log"
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger


def make_output_directory(training_details: TrainingDetails):
    root_dir = get_repo_root_dir()
    final_dir_name = Path(str(training_details)) / f"{datetime.now():%H_%M_%S}"
    final_output_path = root_dir / "models" / final_dir_name
    os.makedirs(final_output_path, exist_ok=True)
    return final_output_path


def check_lists_equal_length(*lists):
    # Get the length of the first list
    first_length = len(lists[0])

    # Check the lengths of the remaining lists
    for lst in lists[1:]:
        if len(lst) != first_length:
            raise ValueError("Lists are not of equal length")




