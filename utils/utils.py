import matplotlib.pyplot as plt
from settings import model_setting
import torch


def post_process_sequence_batch(batch_tuple):

    input_sequences, output_sequences, lengths = batch_tuple

    splitted_input_sequence_batch = input_sequences.split(split_size=1)
    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    splitted_lengths_batch = lengths.split(split_size=1)

    training_data_tuples = zip(splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch)

    training_data_tuples_sorted = sorted(training_data_tuples, key=lambda p: int(p[2]), reverse=True)

    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(
        *training_data_tuples_sorted
    )

    input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
    lengths_batch_sorted = torch.cat(splitted_lengths_batch)

    input_sequence_batch_sorted = input_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0] :, :]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0] :, :]

    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)

    lengths_batch_sorted_list = list(lengths_batch_sorted)
    lengths_batch_sorted_list = map(lambda x: int(x), lengths_batch_sorted_list)

    return input_sequence_batch_transposed, output_sequence_batch_sorted, list(lengths_batch_sorted_list)


def interpolate(
    model, n_interpolations, sequence_length, sos=None, device="cuda" if torch.cuda.is_available() else "cpu"
):

    # # Get input.

    z1 = torch.randn((1, 1, model_setting["latent_size"])).to(device)
    z2 = torch.randn((1, 1, model_setting["latent_size"])).to(device)

    tone1 = model.inference(sequence_length, z1, sos)
    tone2 = model.inference(sequence_length, z2, sos)

    alpha_s = torch.linspace(0, 1, n_interpolations)

    interpolations = torch.stack([alpha * z1 + (1 - alpha) * z2 for alpha in alpha_s])

    samples = [model.inference(sequence_length, z, sos) for z in interpolations]

    samples = torch.stack(samples)

    return samples, tone1, tone2


def plot_elbo(losses, mode):
    elbo_loss = list(map(lambda x: x[0], losses))
    kl_loss = list(map(lambda x: x[1], losses))
    recon_loss = list(map(lambda x: x[2], losses))

    losses = {"elbo": elbo_loss, "kl": kl_loss, "recon": recon_loss}
    print(losses)
    for key in losses.keys():
        plt.plot(losses.get(key), label=key + "_" + mode)

    plt.legend()
    plt.show()
