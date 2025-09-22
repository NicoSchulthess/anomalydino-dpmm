import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.special import digamma
from torch.distributions import MultivariateNormal, Multinomial, Beta
from tqdm import tqdm


def distance(
    x:                  torch.Tensor,
    y:                  torch.Tensor,
    weight_matrix_chol: torch.Tensor | None  = None,
):
    """
    args:
        x: NxD
        y: KxD
        weight_matrix_chol (optional): 1xDxD or KxDxD

    Returns:
    pairwise distance matrix between x and y (size: NxK)
    """

    delta = x[:,None,:] - y[None,:,:]  # NxKxD
    if weight_matrix_chol is not None:
        distance = torch.linalg.norm(torch.einsum("kDd,nkd->nkD", weight_matrix_chol, delta), dim=2)
    else:
        distance = torch.linalg.norm(delta, dim=2)

    return distance


class DPMM:
    def __init__(
            self,
            K           : int,
            D           : int,
            update_rate : float | None  = None,   # 0 <= rate <= 1
            schedule    : str           = "ema",  # "ema": exponentially moving average, "exp": exponentially decaying step size
            alpha       : float         = 2.0,
            alpha_fixed : bool          = False,
            eps         : float | None  = None,
            cov_type    : str           = "full",  # "full", "diag" or "spherical" covariances
            reg_covar   : float         = 1e-6,
            device      : str           = "cpu",
        ):
        assert alpha >= 1, f"alpha needs to be >= 1, but is {alpha}"

        self.K = K  # number of components
        self.D = D  # data dimension
        self.update_rate = update_rate
        self.schedule = schedule.lower()
        self.alpha = alpha  # scaling parameter (higher alpha <-> more components)
        self.alpha_fixed = alpha_fixed
        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps
        self.cov_type = cov_type
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

        beta_distribution = Beta(1, self.alpha)
        breaking_ratios = beta_distribution.sample((self.K,))
        breaking_ratios[-1] = 1
        self.v = breaking_ratios.to(self.device)

        self.mean = torch.randn(self.K, self.D, device=self.device)         # shape: KxD
        if data is None:
            self.cov = torch.eye(self.D, device=self.device)                # shape: DxD
        else:
            data = data.float()
            n_samples = min(self.K, data.shape[0])
            self.mean[:n_samples, :] = data[torch.randperm(n_samples), :]
            self.cov = torch.diag(torch.var(data, dim=0) + self.reg_covar)  # shape: DxD
        self.cov = self.cov[None, ...].repeat(self.K, 1, 1)                 # shape: KxDxD
        self.compute_covariance_cholesky()

    def state_dict(self):
        return {
            "iterations": self.iterations,
            "resp_stat": self.resp_stat,
            "mean_stat": self.mean_stat,
            "cov_stat": self.cov_stat,
            "v": self.v,
            "mean": self.mean,
            "cov": self.cov,
        }
    
    def load_state_dict(self, state_dict):
        self.iterations = state_dict["iterations"]
        self.resp_stat = state_dict["resp_stat"]
        self.mean_stat = state_dict["mean_stat"]
        self.cov_stat = state_dict["cov_stat"]
        self.v = state_dict["v"]
        self.mean = state_dict["mean"]
        self.cov = state_dict["cov"]
        self.compute_covariance_cholesky()

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

    def calculate_pi(self) -> torch.Tensor:
        pi = torch.cumprod(1 - self.v, 0)
        pi = torch.roll(pi, 1)
        pi[0] = 1
        pi *= self.v
        pi /= pi.sum()  # Ensuring that all pi add up to 1.
        return pi
    
    def calculate_log_pi(self) -> torch.Tensor:
        log_pi = torch.cumsum(torch.log(1 - self.v), 0)
        log_pi = torch.roll(log_pi, 1)
        log_pi[0] = 0
        log_pi += torch.log(self.v)
        return log_pi
    
    def get_log_prob(
        self,
        data: torch.Tensor,
        weight_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        args:
            data: Nx1xD

        Returns:
        log(pi_i * N(y_t | mu_i, cov_i)) of size NxK
        """
        log_pi = self.calculate_log_pi()

        component_mask = torch.exp(log_pi) > weight_threshold

        gaussian = MultivariateNormal(
            self.mean[component_mask, :],
            scale_tril=self.cov_chol[component_mask, :, :],
        )
        log_prob = torch.full((data.shape[0], self.K), -torch.inf, device=self.device)
        log_prob[:, component_mask] = gaussian.log_prob(data)
        return log_prob
    
    def get_weighted_log_prob(
        self,
        data: torch.Tensor,
        weight_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        args:
            data: Nx1xD

        Returns:
        log(pi_i * N(y_t | mu_i, cov_i)) of size NxK
        """
        log_pi = self.calculate_log_pi()

        component_mask = torch.exp(log_pi) > weight_threshold

        gaussian = MultivariateNormal(
            self.mean[component_mask, :],
            scale_tril=self.cov_chol[component_mask, :, :],
        )
        weighted_log_prob = torch.full((data.shape[0], self.K), -torch.inf, device=self.device)
        weighted_log_prob[:, component_mask] = log_pi[component_mask] + gaussian.log_prob(data)
        return weighted_log_prob

    def get_responsibilities(self, data: torch.Tensor) -> torch.Tensor:
        """
        args:
            data: Nx1xD

        Returns:
        P(x_t = i | y_t, Phi^{t-1}) of size NxK
        """
        pi = self.calculate_pi()
        gaussian = MultivariateNormal(self.mean, self.cov)
        resp = pi * torch.exp(gaussian.log_prob(data))
        resp /= resp.sum(1, keepdim=True)
        return resp
    
    def get_log_responsibilities(self, data: torch.Tensor) -> torch.Tensor:
        """
        args:
            data: Nx1xD

        Returns:
        log(P(x_t = i | y_t, Phi^{t-1})) of size NxK
        """
        weighted_log_prob = self.get_weighted_log_prob(data)
        log_resp = weighted_log_prob - weighted_log_prob.logsumexp(1, keepdim=True)
        return log_resp
    
    def score(self, data: torch.Tensor) -> torch.Tensor:
        """
        args:
            data: NxD

        Returns:
        log(sum_i( pi_i * N(y_t | mu_i, cov_i))) averaged over all N points of size 1
        """
        score = self.sample_score(data)     # N
        mean_score = score.mean()           # 1
        return mean_score
    
    def sample_score(
        self,
        data: torch.Tensor,
        weight_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        args:
            data: NxD

        Returns:
        log(sum_i( pi_i * N(y_t | mu_i, cov_i))) of size N
        """
        data = data[:,None,:]
        weighted_log_prob = self.get_weighted_log_prob(data, weight_threshold)    # NxK
        score = torch.logsumexp(weighted_log_prob, dim=1)       # N
        return score
    
    def e_step(self, data: torch.Tensor) -> torch.Tensor:
        log_resp = self.get_log_responsibilities(data)  # NxK
        return log_resp
    
    def estimate_covariance_full(self):
        cov = self.cov_stat / (self.resp_stat[:,None,None] + self.eps)
        mean_cov = self.mean[:,None,:] * self.mean[:,None,:].transpose(-2,-1)
        cov -= mean_cov
        cov += torch.eye(self.D, device=self.device)[None,:,:] * self.reg_covar
        return cov
    
    def estimate_covariance_diagonal(self):
        idxs = torch.arange(self.D)
        data_squared = self.cov_stat[:, idxs, idxs] / (self.resp_stat[:,None] + self.eps)
        mean_squared = self.mean ** 2
        mixed_term = self.mean * self.mean_stat / (self.resp_stat[:,None] + self.eps)
        variance = data_squared + mean_squared - 2 * mixed_term
        cov = torch.zeros_like(self.cov)
        cov[:, idxs, idxs] = variance + self.reg_covar
        return cov
    
    def estimate_covariance_spherical(self):
        diagonal_cov = self.estimate_covariance_diagonal()
        variance = diagonal_cov.diagonal(dim1=1, dim2=2).mean(1)
        variance = variance[:, None].repeat(1,self.D)
        cov = torch.diag_embed(variance)
        return cov

    def compute_covariance_cholesky(self):
        if self.cov_type == "full":
            self.cov_chol = torch.linalg.cholesky(self.cov, upper=False)
        elif self.cov_type == "diag" or self.cov_type == "spherical":
            self.cov_chol = torch.sqrt(self.cov)
        else:
            raise NotImplementedError()

    def m_step(self, data: torch.Tensor, resp: torch.Tensor):

        n_samples = data.shape[0]

        # Updating sufficient statistics.
        step_size = self.get_step_size()
        self.resp_stat *= (1 - step_size)
        self.mean_stat *= (1 - step_size)
        self.cov_stat *= (1 - step_size)
        
        data_cov = (data * data.transpose(-2,-1))[:,None,:,:]      # Nx1xDxD
        self.resp_stat += step_size * resp.mean(0)
        self.mean_stat += step_size * (data * resp[:,:,None]).mean(0)
        self.cov_stat  += step_size * torch.einsum("...ndf,nk->kdf", data_cov.transpose(0,1), resp) / n_samples

        # Updating cluster means.
        self.mean = self.mean_stat / (self.resp_stat[:,None] + self.eps)

        # Updating cluster covariances.
        if self.cov_type == "full":
            self.cov = self.estimate_covariance_full()
        elif self.cov_type == "diag":
            self.cov = self.estimate_covariance_diagonal()
        elif self.cov_type == "spherical":
            self.cov = self.estimate_covariance_spherical()
        else:
            raise NotImplementedError()
        self.compute_covariance_cholesky()

        # Updating stick breaking ratios v.
        double_sum = torch.cumsum(self.resp_stat.flip(0), 0).flip(0)
        double_sum[0] = 0
        double_sum = double_sum.roll(-1)
        self.v = self.resp_stat / (self.resp_stat + (self.alpha - 1) / n_samples + double_sum + self.eps)

        # Updating scaling parameter alpha.
        if not self.alpha_fixed:
            self.alpha = (self.K - 1) / (
                digamma(n_samples * (self.resp_stat + double_sum) + self.alpha + 1) -
                digamma(n_samples * double_sum + self.alpha)
            )[:-1].sum().item()

    def step(self, data: torch.Tensor):
        self.iterations += 1
        data = data[:,None,:].float()  # Nx1xD
        log_resp = self.e_step(data)
        self.m_step(data, torch.exp(log_resp))

    def sample(self, n:int=1) -> torch.Tensor:
        samples = []
        
        pi = self.calculate_pi()
        counts = Multinomial(total_count=n, probs=pi).sample()
        
        for i, count in enumerate(counts):
            if count == 0:
                continue
            gaussian = MultivariateNormal(self.mean[i], self.cov[i])
            samples.append(gaussian.sample((count.int().item(),)))
        
        samples = torch.cat(samples).to(self.device)
        return samples
    
    def distance_to_nearest_cluster(
            self,
            data: torch.Tensor,
            weight_threshold: float = 0.0,
            covariance_weighted_norm: bool = False,
            return_min: bool = True,
        ) -> torch.Tensor:
        """
        args:
            data: NxD

        Returns:
        distance of each vector to the nearest cluster (size: N)
        """

        pi = self.calculate_pi()
        component_mask = pi > weight_threshold
        means = self.mean[component_mask, :]

        if covariance_weighted_norm:
            cov_chol = self.cov_chol[component_mask, :, :]
            diff = data[:, None, :] - means[None, :, :]
            distances = _batch_mahalanobis(cov_chol, diff).sqrt()
        else:
            distances = distance(data, means)

        if return_min:
            distances = distances.amin(1)

        return distances
    

    def cosine_distance_to_nearest_cluster(
            self,
            data: torch.Tensor,
            weight_threshold: float = 0.0,
            return_min: bool = True,
        ) -> torch.Tensor:
        """
        args:
            data: NxD

        Returns:
        distance of each vector to the nearest cluster (size: N)
        """

        pi = self.calculate_pi()
        component_mask = pi > weight_threshold
        means = self.mean[component_mask, :]

        distances = 1 - torch.cosine_similarity(data[:,None,:], means[None,:,:], dim=2)

        if return_min:
            distances = distances.amin(1)

        return distances
    
    def visualize(
        self,
        data=None,
        labels=None,
        visualize_model=True,
        visualize_samples=0,            # number of samples to be visualized (if <= 0, ellipses are visualized instead)
        pca : PCA = None,
        ax=None,
        ax_model=None,
        ax_hist=None,
        xlim=None,
        ylim=None,
        pi_cutoff=1e-2,
        normalize_pi_color=False,
        scatter_alpha=0.5,
        cov_color_curve=0.5,
    ):
        """
        Plots the GMM with ellipses representing Gaussians.
        
        Parameters:
            data (torch.Tensor): Tensor of shape (N, D) representing the data points (optional).
            labels (torch.Tensor): Tensor of shape (N,) representing cluster assignments (optional).
            pca (sklearn.decomposition.PCA): Tensor of shape (2, D) representing the two principal components for visualization (optional).
            ax (matplotlib.axes.Axes): Matplotlib Axes to plot on (optional).
        """

        assert self.D == 2 or data is not None or pca is not None, "This function only supports 2D visualizations or higher dimensionality when a PCA object or data is provided."

        if ax is None and ax_hist is None:
            _, (ax, ax_hist) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 8))

        if ax_model is None:
            ax_model = ax

        if data is not None:
            if self.D > 2 and pca is None:
                pca = PCA(n_components=2)
                data = pca.fit_transform(data.cpu().numpy())
            elif self.D > 2 and pca is not None:
                data = pca.transform(data.cpu().numpy())
            else:
                data = data.cpu().numpy()

            if labels is not None:
                labels = labels.cpu().numpy()
                ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=10, alpha=scatter_alpha, vmax=10)
            else:
                ax.scatter(data[:, 0], data[:, 1], s=10, alpha=scatter_alpha)

        if not visualize_model:
            return

        if visualize_samples > 0 and (self.D == 2 or pca is not None):
            # Plot samples from the gaussian mixture model
            samples = self.sample(int(visualize_samples))

            if self.D == 2:
                samples = samples.cpu().numpy()
            else:
                samples = pca.transform(samples.cpu().numpy())
            ax_model.scatter(samples[:, 0], samples[:, 1], s=10, alpha=scatter_alpha)
        else:
            # Plot the Gaussians
            means = self.mean.cpu().numpy()
            covariances = self.cov.cpu().numpy()
            pi = self.calculate_pi().cpu().numpy()

            if ax_hist is not None:
                ax_hist.set_xscale("log")
                bins = 10**np.linspace(np.log10(pi[pi > 0].min()), 0, 51)
                ax_hist.hist(pi, bins=bins, label=f"{(pi > 0).sum()} / {pi.shape[0]} active components")
                ax_hist.grid(visible=True, which="both")
                ax_hist.legend()

            if pca is not None:
                means = pca.transform(means)
                covariances = np.einsum("kd,nde,le->nkl", pca.components_, covariances, pca.components_)
            
            for i in range(self.K):
                if pi[i] < pi_cutoff:
                    continue
                mean = means[i]
                cov = covariances[i]

                # Compute the ellipse parameters
                eigvals, eigvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
                width, height = 4 * np.sqrt(eigvals)  # 4*sqrt(eigvals) gives 2-std ellipse (≃ 95% of the data)

                if normalize_pi_color:
                    pi /= pi.max()
                color = (1, 0, 0, min(1, max(0, pi[i].item())) ** cov_color_curve)

                # Plot the ellipse
                ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', lw=2)
                ax.add_patch(ellipse)
                
                # Mark the mean
                ax.scatter(*mean, color=color, s=50, marker='x')

        ax.set_aspect('equal', adjustable='box')
        ax_model.set_aspect('equal', adjustable='box')
        if xlim is not None:
            ax.set_xlim(xlim)
            ax_model.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            ax_model.set_ylim(ylim)


if __name__ == "__main__":
    from gmm import GMM

    torch.manual_seed(0)
    n_steps = 100
    n_samples = int(1e3)
    dp_K = 100
    update_rate = 0.2
    schedule = "ema"        # "ema" or "exp"
    dp_alpha = 2
    dp_alpha_fixed = False
    cov_type = "spherical"       # "full", "diag" or "spherical"

    
    experiment_id = f"cov_{cov_type}_n_samples_{n_samples}_DP_K_{dp_K}_update_{schedule}{update_rate}_alpha_{dp_alpha}_alpha_fixed_{dp_alpha_fixed}"

    # data_generation_gmm = GMM(K=3, D=2)
    # data_generation_gmm.set_params(
    #     torch.tensor([0.3, 0.3, 0.4]),
    #     torch.tensor([[3, 3], [-3, 3], [0, -3]], dtype=float),
    #     torch.tensor([[[1, -0.5], [-0.5, 1]], [[1, 0.5], [0.5, 1]], [[1, 0], [0, 0.5]]], dtype=float),
    # )

    data_generation_means = torch.tensor([1, 1, 1, 1, 10, 1, 1, 1, 1], dtype=float)
    data_generation_means /= data_generation_means.sum()
    data_generation_gmm = GMM(K=9, D=2)
    data_generation_gmm.set_params(
        data_generation_means,
        torch.tensor([[3, 3], [0, 3], [-3, 3], [3, 0], [0, 0], [-3, 0], [3, -3], [0, -2], [-3, -3]], dtype=float),
        torch.tensor([[[1, -0.5], [-0.5, 1]], [[1, 0.5], [0.5, 1]], [[1, 0], [0, 0.5]]], dtype=float).repeat(3,1,1)*0.25,
    )


    dpmm = DPMM(
        K=dp_K,
        D=2,
        update_rate=update_rate,
        schedule=schedule,
        alpha=dp_alpha,
        alpha_fixed=dp_alpha_fixed,
        cov_type=cov_type,
        reg_covar=1e-5,
    )
    
    samples = data_generation_gmm.sample(n_samples)
    dpmm.initialize(samples)

    pi_history = []
    v_history = []
    alpha_history = []
    step_size_history = []
    prob_history = []

    os.makedirs(f"dpmm/{experiment_id}", exist_ok=True)
    for step in tqdm(range(n_steps)):
        pi_history.append(dpmm.calculate_pi())
        v_history.append(dpmm.v)
        alpha_history.append(dpmm.alpha)


        samples = data_generation_gmm.sample(n_samples)

        dpmm.step(samples)

        dpmm.visualize(samples, xlim=(-8, 8), ylim=(-8, 8))
        plt.savefig(f"dpmm//{experiment_id}/step_{step:03d}.jpg")
        plt.close()
        step_size_history.append(dpmm.get_step_size())
        prob_history.append(dpmm.score(samples))

    pi_history.append(dpmm.calculate_pi())
    v_history.append(dpmm.v)
    alpha_history.append(dpmm.alpha)


    pi_history = torch.stack(pi_history)
    v_history = torch.stack(v_history)
    if True:
        fig, ax = plt.subplots(nrows=4, figsize=(16,10))
        ax[0].plot(pi_history)
        # ax[0].set_ylim(0, 1)
        ax[0].set_ylabel(r"Mixing Coefficients $\pi_i$")
        ax[0].grid(True)
        ax[1].plot(alpha_history)
        ax[1].set_ylabel(r"DP scaling parameter $\alpha$")
        ax[1].grid(True)
        ax[2].plot(step_size_history)
        ax[2].set_ylabel(r"Step size $\gamma_t$")
        ax[2].set_ylim(0, 1)
        ax[2].grid(True)
        ax[3].plot(prob_history)
        ax[3].set_ylabel(r"Log Likelihood")
        ax[3].grid(True)
        plt.tight_layout()
        plt.savefig(f"dpmm/{experiment_id}.jpg")
        plt.show()
