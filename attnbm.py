import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_source", type=str, default="mnist", choices=["mnist", "mat"], help='Dataset source for Step 1: "mnist" or "mat" (default: "mnist")')
parser.add_argument("--mat_path", type=str, default=None, help="Path to pre-processed .mat file with key `images` (used when --data_source mat)")
parser.add_argument("--n_sample", type=int, default=200, help="Number of training samples used for ZCA whitening (default: 200)")
parser.add_argument("--img_size", type=int, default=28, help="Image size (MNIST: 28, default: 28)")
parser.add_argument("--batch_size", type=int, default=5, help="Mini-batch size for training (default: 5)")
parser.add_argument("--n_hidden", type=int, default=900, help="Number of hidden units Nh (default: 900)")
parser.add_argument("--epoch", type=int, default=10000, help="Number of training epochs (default: 10000)")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
parser.add_argument("--bc", action="store_true", help="Turn on bias training for b and c (default: off)")
parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
parser.add_argument("--device", type=str, default="auto", help='Torch device (e.g., "cpu", "cuda", default: auto)')
parser.add_argument("--save_path", type=str, default="attnbm_mnist200.pth", help="Path to save trained model weights (default: attnbm_mnist200.pth)")
parser.add_argument("--load_path", type=str, default=None, help="Path to load trained model weights for --skip_train")
parser.add_argument("--skip_train", action="store_true", help="Skip training in Step 2 and load model from --load_path or --save_path")
parser.add_argument("--skip_reconstruct", action="store_true", help="Skip Step 3 (reconstruction and receptive field visualization)")
parser.add_argument("--n_recon", type=int, default=10, help="Number of images used for reconstruction demo (default: 10)")
parser.add_argument("--mask_threshold", type=float, default=0.2, help="Mask threshold used in Step 3 (default: 0.2)")
parser.add_argument("--recon_block", type=int, default=1, help="Block index multiplier k (default: 1)")
parser.add_argument("--n_recep", type=int, default=6, help="Grid size for receptive fields (n_recep x n_recep, default: 6)")
parser.add_argument("--no_plot", action="store_true", help="Disable all matplotlib plots")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def check_images(x, title=None, size=28, n=10):
    n = min(n, len(x))
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x[i].reshape(size, size), cmap="gray")
        plt.title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def zca_whitening(X: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Input: X: M x N matrix; M observations, N variables
    Output: ZCA whitened X: (M, N)
    """
    sigma = np.cov(X, bias=True)  # M x M covariance matrix: (X-mu)*(X-mu)^T / M
    U, S, _ = np.linalg.svd(sigma)

    S2 = np.diag(1.0 / np.sqrt(S + epsilon))
    ZCAMatrix = U @ S2 @ U.T
    return ZCAMatrix @ X


def load_and_preprocess_data(args, device: torch.device):
    img_size2 = args.img_size ** 2

    if args.data_source == "mnist":
        try:
            from keras.datasets import mnist
        except ImportError as exc:
            raise ImportError(
                'Failed to import `keras.datasets.mnist`. Install keras/tensorflow or use "--data_source mat".'
            ) from exc

        (x_train, _), (_, _) = mnist.load_data()
        x_train = x_train.reshape(len(x_train), -1)  # (60000, 28, 28) -> (60000, 784)
        x_train = x_train.astype("float32")
        x_train /= 255.0

        if not args.no_plot:
            print(f"x_train.shape: {x_train.shape}")
            check_images(x_train, title="original", size=args.img_size)

        n_sample = min(args.n_sample, len(x_train))
        x_zca = zca_whitening(x_train[:n_sample])
        x_zca = x_zca / np.std(x_zca)
        x_data = x_zca.astype("float32")

        if not args.no_plot:
            print(f"x_data.shape: {x_data.shape}")
            check_images(x_data, title="ZCA whitened", size=args.img_size)
    else:
        if args.mat_path is None:
            raise ValueError("--mat_path is required when --data_source mat")

        import scipy.io

        mat = scipy.io.loadmat(args.mat_path)
        data_mnist = mat["images"].T
        x_data = (data_mnist / np.std(data_mnist)).astype("float32")
        if x_data.shape[1] != img_size2:
            raise ValueError(
                f"Loaded data dimension is {x_data.shape[1]}, expected {img_size2} (= img_size^2)."
            )

        if not args.no_plot:
            print(f"x_data.shape: {x_data.shape}")
            check_images(x_data, title="original", size=args.img_size)

    # Prepare for training in PyTorch
    X_train = torch.tensor(x_data, dtype=torch.float32, device=device)
    Y_train = torch.tensor(np.empty(x_data.shape[0]), dtype=torch.float32, device=device)
    ds_train = TensorDataset(X_train, Y_train)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    return x_data, X_train, loader_train


class AttnBM(nn.Module):
    """AttnBM in PyTorch.
    Attributes:
        Nv : n_visible
        Nh : n_hidden
        w : weight
        b : bias for visibles
        c : bias for hiddens
        bc : turn on/off b & c (bool, default: False)
        lr : learning rate
    """
    def __init__(self, Nv=None, Nh=900, epoch=10, lr=0.01, bc=False, seed=0, device="cpu"):
        super().__init__()
        torch.manual_seed(seed)

        self.Nv = Nv
        self.Nh = Nh
        self.device = torch.device(device)
        self.w = Parameter(torch.empty((Nh, Nv), device=self.device).normal_(mean=0, std=1) / np.sqrt(Nv))
        self.b = Parameter(torch.zeros(1, Nv, device=self.device))
        self.c = Parameter(torch.zeros(1, Nh, device=self.device))
        self.epoch = epoch
        self.lr = lr

        self.params = [self.w]
        if bc:
            self.params.extend([self.b, self.c])

        self.optim = optim.SGD(self.params, lr=self.lr)
        self.losses = []
        self.bs = []
        self.cs = []

    # two-layer feedforward MLP: the update rule of the model B, see Eq.13
    def forward(self, v):
        h = F.linear(v, self.w)
        soft = F.softmax(h, dim=-1)
        v_out = F.linear(soft, self.w.T)
        return v, v_out

    # image reconstruction, see Eq.39
    def reconstruct(self, v_o, missing, n_recon=10):
        v_out = torch.empty((n_recon, self.Nv), device=self.device)
        for i in range(n_recon):
            w_m = self.w * missing[i]
            temp = F.linear(v_o[i], self.w) + 0.5 * torch.square(w_m).sum(dim=-1)
            soft = F.softmax(temp, dim=-1)
            v_m = F.linear(soft, w_m.T)
            v_out[i] = v_o[i] + v_m
        return v_o, v_out

    # display receptive fields
    def get_recep(self, n_recep=6):
        size = int(np.sqrt(self.Nv))
        plt.figure(figsize=(20, 12))
        for i in range(n_recep ** 2):
            z = self.w[i, :].view(size, size).detach().cpu().numpy()
            plt.subplot(n_recep, n_recep, i + 1)
            plt.imshow(z, cmap="gray")
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def loss(self, v):
        energy = 0.5 * torch.square(v - self.b).sum(dim=-1) - torch.logsumexp(F.linear(v, self.w, self.c), dim=-1)
        logz = torch.logsumexp(0.5 * torch.square(self.w).sum(dim=-1) + F.linear(self.b, self.w, self.c), dim=-1)
        return energy + logz

    def fit(self, train_loader):
        for epoch in tqdm(range(self.epoch)):
            start = time.time()
            train_loss_epoch = 0.0

            for data, _ in train_loader:
                train_loss = self.loss(data).mean()
                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()
                train_loss_epoch += train_loss.item()

            temp = train_loss_epoch / len(train_loader.dataset)
            self.losses.append(temp)
            self.bs.append(self.b.norm().item())
            self.cs.append(self.c.norm().item())
            end = time.time()
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss = {temp:.3f}, {end - start:.2f}s")


def create_model(args, device: torch.device):
    img_size2 = args.img_size ** 2
    return AttnBM(
        Nv=img_size2,
        Nh=args.n_hidden,
        epoch=args.epoch,
        lr=args.lr,
        bc=args.bc,
        seed=args.seed,
        device=str(device),
    )


def train_or_load_model(args, loader_train, device: torch.device):
    model = create_model(args, device)
    if args.skip_train:
        load_path = args.load_path if args.load_path is not None else args.save_path
        state = torch.load(load_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded model from: {load_path}")
        return model

    model.fit(loader_train)
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved model to: {args.save_path}")
    return model


def plot_training_curves(model):
    plt.plot(model.losses)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.show()


def vis_recep(model, n_recep=6):
    size = int(np.sqrt(model.Nv))
    plt.figure(figsize=(12, 12))
    sortidxs = np.argsort(np.linalg.norm(model.w.detach().cpu().numpy(), axis=-1))
    for i in range(n_recep ** 2):
        z = model.w[sortidxs]
        z = z[-(i + 1), :].view(size, size).detach().cpu().numpy()
        plt.subplot(n_recep, n_recep, i + 1)
        plt.imshow(z, cmap="gray")
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def run_reconstruction_demo(args, model, X_train, x_data):
    n = args.n_recon
    img_size2 = args.img_size ** 2

    mask = (np.random.rand(n, img_size2) > args.mask_threshold).astype(int)
    missing = torch.tensor(mask, dtype=torch.float32, device=model.device)
    observed = torch.tensor(np.ones([n, img_size2]) - mask, dtype=torch.float32, device=model.device)

    nk = n * args.recon_block
    if nk + n > X_train.shape[0]:
        raise ValueError(f"Requested slice [{nk}:{nk + n}] exceeds dataset size {X_train.shape[0]}.")

    v = X_train[nk : nk + n].to(model.device)
    v_o = v * observed

    v_mask = v_o.detach().cpu().numpy()
    v_recon = model.reconstruct(v_o, missing, n_recon=n)[1].detach().cpu().numpy()
    v_out = model(v_o)[1].detach().cpu().numpy()

    if not args.no_plot:
        check_images(x_data[nk : nk + n], "original", size=args.img_size, n=n)
        check_images(v_mask, "masked", size=args.img_size, n=n)
        check_images(v_recon, "reconstructed", size=args.img_size, n=n)
        check_images(v_out, "single update", size=args.img_size, n=n)

        print("\nReceptive fields:")
        model.get_recep(n_recep=args.n_recep)
        vis_recep(model, n_recep=args.n_recep)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    # (1/3): Pre-process data (ZCA whitening or load pre-processed data)
    x_data, X_train, loader_train = load_and_preprocess_data(args, device)

    # (2/3): Define and train/load AttnBM
    model = train_or_load_model(args, loader_train, device)

    if not args.no_plot and not args.skip_train:
        plot_training_curves(model)

    # (3/3): Image reconstruction and receptive field visualization
    if not args.skip_reconstruct:
        run_reconstruction_demo(args, model, X_train, x_data)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
