from models import Discriminator
import torch
from datetime import datetime, timedelta

from batcher import Batcher


def train():
    loader = Batcher(batch_size=128)

    disc = Discriminator(num_glimpses=4, lstm_out=128)
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=disc.parameters(), lr=3e-4)

    best_validation_loss = None
    saving_threshold = 1.2
    last_saved = None
    save_every = timedelta(minutes=10)


    i = -1
    while True:
        i += 1

        X, Y = loader.fetch_batch("train")
        pred = disc(X)
        loss = bce(pred, Y.float())

        if i % 10 == 0:

            # validate your model
            X_val, Y_val = loader.fetch_batch("val")
            pred_val = disc(X_val)
            loss_val = bce(pred_val, Y_val.float())

            training_loss = loss.data[0]
            validation_loss = loss_val.data[0]

            print("Training loss: {} \t Validation loss: {}".format(
                training_loss, validation_loss
            ))

            if best_validation_loss is None or best_validation_loss > (saving_threshold * validation_loss):
                print("Significantly improved validation loss from {} --> {}. Saving...".format(
                    best_validation_loss, validation_loss
                ))
                best_validation_loss = validation_loss
                torch.save(disc.state_dict(), "saved_models/disc-{}".format(best_validation_loss))
                last_saved = datetime.utcnow()

            if last_saved is None or last_saved + save_every < datetime.utcnow():
                print("It's been too long since we last saved the model. Saving...")
                torch.save(disc.state_dict(), "saved_models/disc-{}".format(validation_loss))
                last_saved = datetime.utcnow()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main() -> None:
    train()


if __name__ == "__main__":
    main()