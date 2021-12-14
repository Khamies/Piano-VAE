import torch
from utils.utils import post_process_sequence_batch


class Trainer:

    def __init__(self, train_loader, test_loader, model, loss, optimizer) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.interval = 10


    def train(self, train_losses, epoch, batch_size, clip) -> list:  
            # Initialization of RNN hidden, and cell states.

        states = self.model.init_hidden(batch_size) 
        interval = 10

        for batch_num, batch in enumerate(self.train_loader):
            # get the labels
            source, target, source_lengths = post_process_sequence_batch(batch)
            source = source.reshape(source.size(1), source.size(0), source.size(2)).to(self.device)
            target = target.to(self.device)
            source_lengths = torch.tensor(source_lengths)


            x_hat_param, mu, log_var, z, states = self.model(source,source_lengths, states)

            # Detach hidden states
            states = states[0].detach(), states[1].detach()

            # Compute the loss
            mloss, KL_loss, recon_loss = self.loss(mu = mu, log_var = log_var, z = z, x_hat_param = x_hat_param , x = target)

            train_losses.append((mloss , KL_loss.item(), recon_loss.item()))
            
            # Backward the loss
            mloss.backward()

            # Clip the gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()

            self.optimizer.zero_grad()



            if batch_num % self.interval == 0 :
        
                print('| epoch {:3d} | {:5d}/{:5d} batches | elbo_loss {:5.6f} | kl_loss {:5.6f} | recons_loss {:5.6f} '.format(
                    epoch, batch_num, len(self.train_loader), mloss.item(), KL_loss.item(), recon_loss.item()))
            
        return train_losses


    def test(self, test_losses, epoch, batch_size) -> list:

        with torch.no_grad():

            states = self.model.init_hidden(batch_size) 

            for batch_num, batch in enumerate(self.test_loader): # loop over the data, and jump with step = bptt.
                # get the labels
                source, target, source_lengths = post_process_sequence_batch(batch)
                source = source.reshape(source.size(1), source.size(0), source.size(2)).to(self.device)
                target = target.to(self.device)
                source_lengths = torch.tensor(source_lengths)



                x_hat_param, mu, log_var, z, states = self.model(source,source_lengths, states)

                # detach hidden states
                states = states[0].detach(), states[1].detach()

                # compute the loss
                mloss, KL_loss, recon_loss = self.loss(mu = mu, log_var = log_var, z = z, x_hat_param = x_hat_param , x = target)

                test_losses.append((mloss , KL_loss.item(), recon_loss.item()))

                #Statistics.
                if batch_num % self.interval ==0:
                  print('| epoch {:3d} | elbo_loss {:5.6f} | kl_loss {:5.6f} | recons_loss {:5.6f} '.format(
                        epoch, mloss.item(), KL_loss.item(), recon_loss.item()))

            return test_losses