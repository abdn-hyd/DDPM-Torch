import torch


class DDPMScheduler:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "linear",
        dtype: torch.dtype = torch.float64,
    ):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.dtype = dtype

        # create the scheduler
        self.betas = self.get_betas(schedule_type)
        assert isinstance(self.betas, torch.Tensor) and self.betas.dtype == self.dtype
        assert (self.betas > 0).all() and (self.betas <= 1).all()
        self.alphas = 1.0 - self.betas

        # cumulative product of alphas for one step noising for each different steps
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:]]
        )

    def _warmup_beta(
        self,
        warmup_frac,
    ):
        betas = self.beta_end * torch.ones(self.timesteps, dtype=self.dtype)
        warmup_time = int(self.timesteps * warmup_frac)
        betas[:warmup_time] = torch.linspace(
            self.beta_start, self.beta_end, warmup_time, dtype=self.dtype
        )
        return betas

    def get_betas(
        self,
        schedule_type: str,
    ):
        if schedule_type == "linear":
            betas = torch.linspace(
                self.beta_start, self.beta_end, self.timesteps, dtype=self.dtype
            )
        elif schedule_type == "quad":
            betas = torch.linspace(
                self.beta_start**0.5,
                self.beta_end**0.5,
                self.timesteps,
                dtype=self.dtype,
            )
        elif schedule_type == "const":
            betas = self.beta_end * torch.ones(self.timesteps, dtype=self.dtype)
        elif schedule_type == "jsd":
            betas = torch.linspace(self.timesteps, 1, self.timesteps, dtype=self.dtype)
        elif schedule_type == "warmup10":
            betas = self._warmup_beta(0.1)
        elif schedule_type == "warmup50":
            betas = self._warmup_beta(0.5)
        else:
            raise NotImplementedError("Incorrect schedule type!")
        assert betas.shape == (self.timesteps,)
        return betas

    def forward_process(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        device = x.device
        sqrt_alphas_t = torch.sqrt(self.alphas_cumprod[t].view(-1, 1, 1, 1).to(device))
        sqrt_one_minus_alphas_t = torch.sqrt(
            1.0 - self.alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        )

        # generate noise if noise is none
        if not noise:
            noise = torch.randn_like(x).to(device)

        return sqrt_alphas_t * x + sqrt_one_minus_alphas_t * noise, noise

    def sample_prev(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        device = xt.device

        # coefficients for sampling
        alphas_t = self.alphas[t].view(-1, 1, 1, 1).to(device)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        alphas_cumprod_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1).to(device)
        beta_t = self.betas[t].view(-1, 1, 1, 1).to(device)

        # using current predicted noise to predict the final image
        x0 = (xt - (torch.sqrt(1.0 - alphas_cumprod_t) * noise_pred)) / torch.sqrt(
            alphas_cumprod_t
        )
        x0 = torch.clamp(x0, -1.0, 1.0)

        # the mean of predicted image using predicted noise
        """
        based on this equation, we can infer that the mean of image is only associated with the predicted noise,
        and this is why the model will only predict the noise instead of predicting image of specific time step
        """
        mean = (
            xt - (beta_t / torch.sqrt(1 - alphas_cumprod_t)) * noise_pred
        ) / torch.sqrt(alphas_t)

        # using sigma and gaussian noise to add diversity of generation process
        variance = beta_t * (1.0 - alphas_cumprod_t_prev) / (1.0 - alphas_cumprod_t)
        sigma = torch.sqrt(variance)
        if not noise:
            noise = torch.randn_like(xt).to(device)

        # nonzero mask will indentify whether it is the last time step or not, to avoid adding noise
        nonzero_mask = (t > 0).float().view(-1, 1, 1, 1).to(device)
        xt_prev = mean + nonzero_mask * sigma * noise

        return xt_prev, x0


if __name__ == "__main__":
    scheduler = DDPMScheduler()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    x_noised, noise = scheduler.forward_process(x, t)
    print(x_noised.shape, noise.shape)
    x_prev, x0 = scheduler.sample_prev(x_noised, noise, t)
    print(x_prev.shape, x0.shape)
