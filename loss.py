"""
Calculate Loss functions
"""
# pylint: disable=no-member
import torch


class VAELoss(torch.nn.Module):
    """
    Calculate VAE Loss
    """

    def __init__(self):
        super().__init__()

        self.nlloss = torch.nn.NLLLoss()

    @staticmethod
    def kl_loss(mean, log_var):
        """
        calculate KL loss
        """
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_loss = kl_loss.sum(
            -1
        )  # to go from multi-dimensional z to single dimensional z : (batch_size x latent_size) ---> (batch_size)
        # i.e Z = [ [z1_1, z1_2 , ...., z1_lt] ] ------> z = [ z1]
        #         [ [z2_1, z2_2, ....., z2_lt] ]             [ z2]
        #                   .                                [ . ]
        #                   .                                [ . ]
        #         [[zn_1, zn_2, ....., zn_lt] ]              [ zn]

        #        lt=latent_size
        kl_loss = kl_loss.mean()

        return kl_loss

    def reconstruction_loss(self, x_hat_param, x_input):
        """
        Calculate reconstruction loss
        """
        x_input = x_input.reshape(-1)

        recon = self.nlloss(x_hat_param, input)

        return recon

    def forward(self, mean, log_var, x_hat_param, x_input):
        """
        forward loop
        """
        kl_loss = self.kl_loss(mean, log_var)
        recon_loss = self.reconstruction_loss(x_hat_param, x_input)

        elbo = kl_loss + recon_loss
        # we use + because recon loss is a NLLoss (cross entropy) and it's negative in its own, and in the
        # ELBO equation we have
        # elbo = KL_loss - recon_loss, therefore, ELBO = KL_loss - (NLLoss) = KL_loss + NLLoss

        return elbo, kl_loss, recon_loss
