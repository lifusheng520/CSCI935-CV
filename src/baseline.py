"""Vanilla MobileNetV2 baseline for five-class rice disease classification."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from data_pipeline import RANDOM_SEEDS, prepare_dataloaders

# config
NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 3


class MobileNetV2Baseline(nn.Module):
    """ImageNet-pretrained MobileNetV2 with a five-class output layer."""

    def __init__(self, num_classes: int = NUM_CLASSES, use_pretrained: bool = True) -> None:
        super().__init__()

        # use ImageNet weights for transfer learning
        # passing None creates the same architecture with randomly initialised weights
        weights = (
            MobileNet_V2_Weights.DEFAULT
            if use_pretrained
            else None
        )

        self.model = mobilenet_v2(weights=weights)

        # the original model predicts 1,000 ImageNet classes
        # replace only its final linear layer so that it predicts the five rice disease classes
        input_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one set of five class scores for each input image."""
        return self.model(images)


def train_one_epoch( model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                     device: torch.device) -> tuple[float, float]:
    """train the model for one epoch and return loss and accuracy."""
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:
        # move the current batch to the CPU, Apple GPU, or CUDA GPU
        images = images.to(device)
        labels = labels.to(device)

        # clear gradients left by the previous batch.
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        # calculate gradients and update all trainable model parameters
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        predictions = outputs.argmax(dim=1)
        correct_predictions += (
            predictions == labels
        ).sum().item()

    if total_images == 0:
        raise ValueError("The training DataLoader is empty")

    average_loss = total_loss / total_images
    accuracy = 100.0 * correct_predictions / total_images

    return average_loss, accuracy


def evaluate( model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate the model without updating its parameters."""
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0


    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

            predictions = outputs.argmax(dim=1)
            correct_predictions += (
                predictions == labels
            ).sum().item()

    if total_images == 0:
        raise ValueError("The evaluation DataLoader is empty")

    average_loss = total_loss / total_images
    accuracy = 100.0 * correct_predictions / total_images

    return average_loss, accuracy


def train_model( model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, criterion: nn.Module,
                 optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int,
                 patience: int = EARLY_STOPPING_PATIENCE) -> list[dict]:
    """Train for multiple epochs and restore the best validation model."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than zero")
    if patience <= 0:
        raise ValueError("patience must be greater than zero")

    best_validation_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        # keep the metrics for later plots and experimental reporting.
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss: {train_loss:.4f}, "
            f"train accuracy: {train_accuracy:.2f}% | "
            f"validation loss: {validation_loss:.4f}, "
            f"validation accuracy: {validation_accuracy:.2f}%"
        )

        # save a separate copy because model parameters change every epoch
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # stop when validation loss has not improved for `patience` epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # continue with the parameters from the best validation epoch.
    model.load_state_dict(best_model_state)

    return history


def set_random_seed(random_seed: int) -> None:
    """Set the random generators."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """Select CUDA, Apple Metal, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    # choose CUDA/MPS/CPU
    device = select_device()
    print(f"Using device: {device}\n")

    test_accuracies = {
        "Mixed": [],
        "White": [],
        "Field": [],
    }

    # iterate five seeds
    for random_seed in RANDOM_SEEDS:
        print(f"===== Baseline run: seed {random_seed} =====")
        set_random_seed(random_seed)

        # data loader
        (
            train_loader,
            validation_loader,
            mixed_test_loader,
            white_test_loader,
            field_test_loader,
        ) = (
            prepare_dataloaders(
                random_seed=random_seed,
                batch_size=BATCH_SIZE,
            )
        )

        # each seed must start with a newly initialised model and optimizer
        model = MobileNetV2Baseline(
            num_classes=NUM_CLASSES,
            use_pretrained=True,
        ).to(device)
        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        # train for the configured number of epochs and early stopping
        train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=NUM_EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
        )

        # evaluate the best model on all three required test scenarios.
        mixed_test_loss, mixed_test_accuracy = evaluate(
            model=model,
            data_loader=mixed_test_loader,
            criterion=criterion,
            device=device,
        )
        white_test_loss, white_test_accuracy = evaluate(
            model=model,
            data_loader=white_test_loader,
            criterion=criterion,
            device=device,
        )
        field_test_loss, field_test_accuracy = evaluate(
            model=model,
            data_loader=field_test_loader,
            criterion=criterion,
            device=device,
        )

        test_accuracies["Mixed"].append(mixed_test_accuracy)
        test_accuracies["White"].append(white_test_accuracy)
        test_accuracies["Field"].append(field_test_accuracy)

        print(
            f"Seed {random_seed} Mixed: "
            f"loss={mixed_test_loss:.4f}, "
            f"accuracy={mixed_test_accuracy:.2f}%"
        )
        print(
            f"Seed {random_seed} White: "
            f"loss={white_test_loss:.4f}, "
            f"accuracy={white_test_accuracy:.2f}%"
        )
        print(
            f"Seed {random_seed} Field: "
            f"loss={field_test_loss:.4f}, "
            f"accuracy={field_test_accuracy:.2f}%\n"
        )

    print("===== Accuracy over five runs =====")
    for scenario in ("White", "Field", "Mixed"):
        accuracies = test_accuracies[scenario]
        mean_accuracy = np.mean(accuracies)
        standard_deviation = np.std(accuracies)

        print(
            f"{scenario}: "
            f"{mean_accuracy:.2f}% ± {standard_deviation:.2f}%"
        )


if __name__ == "__main__":
    main()
