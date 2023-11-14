import wandb
from trainer import train

if __name__ == "__name__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="2412-Attack Model",

        # track hyperparameters and run metadata
        config={
            "generator_learning_rate": 0.0002,
            "discriminator_learning_rate": 0.0002,
            "architecture": "DCGAN",
            "dataset": "CIFAR-10",
            "optimizer": "Adam",
            "loss function": "Binary Cross Entropy",
            "epochs": 50,
        }
    )

    train(wandb)

    wandb.finish()
