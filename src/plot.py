import matplotlib.pyplot as plt
from pathlib import Path
from NetworkDetails import TrainingDetails
from extras import get_repo_root_dir, check_lists_equal_length
from typing import List
from ast import literal_eval


def load_data(file_path: Path):
    file_name = "network_details.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Can't find file \"{file_name}\" in directory: \"{file_path}\"")

    validation_loss = []
    validation_accuracy = []
    training_loss = []
    training_accuracy = []

    with open(file_path / file_name, "r") as file:
        validation_loss_line = 0
        validation_accuracy_line = 1
        training_loss_line = 2
        training_accuracy_line = 3
        current_line = 0

        for line in file:
            if current_line == validation_loss_line:
                validation_loss = literal_eval(line.strip())
            elif current_line == validation_accuracy_line:
                validation_accuracy = literal_eval(line.strip())
            elif current_line == training_loss_line:
                training_loss = literal_eval(line.strip())
            elif current_line == training_accuracy_line:
                training_accuracy = literal_eval(line.strip())
            else:
                break
            current_line += 1
    return validation_accuracy, validation_loss, training_accuracy, training_loss, file_path


def plot_data(file_path: Path):
    validation_accuracy, validation_loss, training_accuracy, training_loss, filepath = load_data(file_path)
    check_lists_equal_length(validation_accuracy, validation_loss, training_accuracy, training_loss)
    epochs = range(1, len(validation_accuracy) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, training_loss, label="Training")
    ax1.plot(epochs, validation_loss, label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid("on")
    ax1.legend()

    ax2.plot(epochs, training_accuracy, label="Training")
    ax2.plot(epochs, validation_accuracy, label="Validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid("on")
    ax2.legend()

    plt.tight_layout()

    plt.savefig(filepath / "accuracy_validation_plot.png")
    plt.show()


def main():
    training_details = TrainingDetails(batch_size=4,
                                       learning_rate=0.0004,
                                       momentum=0.9,
                                       epochs=2,
                                       output_dir=Path(""))

    file_path = get_repo_root_dir() / "models" / str(training_details) /  "15_48_13"
    plot_data(file_path=file_path)


if __name__ == "__main__":
    main()
