import argparse
import torch
from dataset import PianoGenerationDataset
from model import LSTM_MUSIC_VAE
from loss import VAELoss
from train import Trainer
from settings import global_setting, training_setting, model_setting

# General Settings

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(global_setting["seed"])


parser = argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")
parser.add_argument("--batch_size", type=str, default="32")
parser.add_argument("--bptt", type=str, default="60")
parser.add_argument("--embed_size", type=str, default="300")
parser.add_argument("--hidden_size", type=str, default="256")
parser.add_argument("--latent_size", type=str, default="16")
parser.add_argument("--lr", type=str, default="0.001")


# Extract commandline arguments
args = parser.parse_args()

batch_size = int(args.batch_size) if args.batch_size != None else training_setting["batch_size"]
bptt = int(args.bptt) if args.bptt != None else training_setting["bptt"]
embed_size = int(args.embed_size) if args.embed_size != None else training_setting["embed_size"]
hidden_size = int(args.hidden_size) if args.hidden_size != None else training_setting["hidden_size"]
latent_size = int(args.latent_size) if args.latent_size != None else training_setting["latent_size"]
lr = float(args.lr) if args.lr != None else training_setting["lr"]
lstm_layer = model_setting["lstm_layer"]


# Load the data
trainset = PianoGenerationDataset(
    "./data/Nottingham/train/", longest_sequence_length=None
)  #  Instead of None, use : model_setting["bptt"] to control the sequence length.
testset = PianoGenerationDataset("./data/Nottingham/test/", longest_sequence_length=None)


# Batchify the data
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)


vocab_size = model_setting["note_size"]

model = LSTM_MUSIC_VAE(
    vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size, latent_size=latent_size
).to(device)

Loss = Loss = VAELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

trainer = Trainer(train_loader, test_loader, model, Loss, optimizer)

if __name__ == "__main__":

    # Epochs
    train_losses = []
    test_losses = []
    for epoch in range(training_setting["epochs"]):
        print("Epoch: ", epoch)
        print("Training.......")
        train_losses = trainer.train(train_losses, epoch, training_setting["batch_size"], training_setting["clip"])
        # print("Testing.......")
        # test_losses = trainer.test(test_losses, epoch, training_setting["batch_size"])

    # plot_elbo(train_losses, "train")
    # plot_elbo(test_losses, "test")
