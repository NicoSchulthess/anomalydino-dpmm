import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Multinomial




class GMM:
    def __init__(
        self,
        K               : int,
        D               : int,
        update_rate     : float | None = None,
        schedule        : str          = "ema",
        eps             : float | None = None,
        cov_type        : str          = "full",
        reg_covar       : float        = 1e-6,
        device          : str          = "cpu",
    ):
        self.K = K  # number of components
        self.D = D  # data dimension
        self.reg_covar = reg_covar

        device = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.initialize()

    def initialize(self, data=None):
        self.iterations = 0

        # History of sufficient statistics.
        self.resp_stat = torch.zeros((self.K,), device=self.device)                 # shape: K
        self.mean_stat = torch.zeros((self.K, self.D), device=self.device)          # shape: KxD
        self.cov_stat  = torch.zeros((self.K, self.D, self.D), device=self.device)  # shape: KxDxD

        self.pi = torch.ones(self.K, device=self.device) / self.K           # shape: K
        self.mean = torch.randn(self.K, self.D, device=self.device)         # shape: KxD
        if data is None:
            self.cov = torch.eye(self.D, device=self.device)                # shape: DxD
        else:
            data = data.float()
            n_samples = min(self.K, data.shape[0])
            self.mean[:n_samples, :] = data[torch.randperm(n_samples), :]
            self.cov = torch.diag(torch.var(data, dim=0))                   # shape: DxD
        self.cov = self.cov[None, ...].repeat(self.K, 1, 1)                 # shape: KxDxD
        self.compute_covariance_cholesky()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self):
        raise NotImplementedError()

    def get_step_size(self) -> float:
        if self.update_rate is None:
            # No history is used, corresponds to "standard" expectation maximization
            return 1

        if self.schedule == "ema":
            step_size = self.update_rate
        elif self.schedule == "exp":
            step_size = self.iterations ** (-self.update_rate)
        else:
            raise NotImplementedError()
        return step_size

    def get_cluster_prob(self, data: torch.Tensor) -> torch.Tensor:
        """
        args:
            data: Nx1xD

        Returns:
        P(x_t = i | y_t, Phi^{t-1}) of size NxK
        """
        gaussian = MultivariateNormal(self.mean, self.cov)
        cluster_prob = self.pi * torch.exp(gaussian.log_prob(data))
        cluster_prob /= cluster_prob.sum(1, keepdim=True)
        return cluster_prob

    def em_step(self, data):
        if self.algorithm == "batch_em":
            data = data[:,None,:]  # Nx1xD

            cluster_prob = self.get_cluster_prob(data)
            cluster_prob_mean = cluster_prob.mean(0)

            # Update of GMM parameters
            self.pi = cluster_prob_mean
            self.mean = (data * cluster_prob[:,:,None]).mean(0) / cluster_prob_mean[:,None]
            data_cov = (data * data.transpose(-2,-1))[:,None,:,:]      # Nx1xDxD
            self.cov = (data_cov * cluster_prob[:,:,None,None]).mean(0) / cluster_prob_mean[:,None,None]
            mean_cov = self.mean[:,None,:] * self.mean[:,None,:].transpose(-2,-1)
            self.cov -= mean_cov
            self.cov += torch.eye(self.D) * self.reg_covar

        elif self.algorithm == "recursive_em":
            raise NotImplementedError()

        else:
            raise NotImplementedError(f"GMM.algorithm should be \"batch_em\" or \"recursive_em\", but is {self.algorithm}.")

    def set_params(self, pi=None, mean=None, cov=None):
        if pi is not None:
            self.pi = pi
        if mean is not None:
            self.mean = mean
        if cov is not None:
            self.cov = cov

    def sample(self, n=1):
        samples = []
        counts = Multinomial(total_count=n, probs=self.pi).sample()
        for i, count in enumerate(counts):
            if count == 0:
                continue
            gaussian = MultivariateNormal(self.mean[i], self.cov[i])
            samples.append(gaussian.sample((count.int().item(),)))
        
        samples = torch.cat(samples).to(self.device)
        return samples
    
    def visualize(self, data=None, labels=None, ax=None):
        """
        Plots the GMM with ellipses representing Gaussians.
        
        Parameters:
            means (torch.Tensor): Tensor of shape (K, D) representing the means of the Gaussians.
            covariances (torch.Tensor): Tensor of shape (K, D, D) representing the covariance matrices.
            data (torch.Tensor): Tensor of shape (N, D) representing the data points (optional).
            labels (torch.Tensor): Tensor of shape (N,) representing cluster assignments (optional).
            ax (matplotlib.axes.Axes): Matplotlib Axes to plot on (optional).
        """

        assert self.D == 2, "This function only supports 2D visualizations."

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        if data is not None:
            data = data.cpu().numpy()
            if labels is not None:
                labels = labels.cpu().numpy()
                scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10, alpha=0.5)
                plt.colorbar(scatter, ax=ax, label='Cluster')
            else:
                ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5)

        # Plot the Gaussians
        means = self.mean.cpu().numpy()
        covariances = self.cov.cpu().numpy()

        for i in range(self.K):
            mean = means[i]
            cov = covariances[i]

            # Compute the ellipse parameters
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)  # 2*sqrt(eigvals) gives 1-std ellipse

            color = (1, 0, 0, self.pi[i].item())

            # Plot the ellipse
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', lw=2)
            ax.add_patch(ellipse)
            
            # Mark the mean
            ax.scatter(*mean, color=color, s=50, marker='x')

        plt.axis('equal')



if __name__ == "__main__":

    data_generation_gmm = GMM(K=3, D=2)
    data_generation_gmm.set_params(
        torch.tensor([0.25, 0.25, 0.5]),
        torch.tensor([[-3, -3], [3, -3], [3, 3]], dtype=float),
        torch.tensor([[[1, 0], [0, 1]]], dtype=float).repeat(3,1,1),
    )

    samples = data_generation_gmm.sample(100)

    gmm = GMM(K=10, D=2)

    os.makedirs("gmm", exist_ok=True)
    for step in range(40):
        gmm.em_step(samples)
        gmm.visualize(samples)
        # plt.show()
        plt.savefig(f"gmm/step_{step:02d}.jpg")
        plt.close()
