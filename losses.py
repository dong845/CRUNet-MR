import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SSIM(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, Xt: torch.Tensor, Yt: torch.Tensor, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)
        Xt = (Xt / Xt.max()).unsqueeze(2)
        Yt = (Yt / Yt.max()).unsqueeze(2)
        ssims = 0.0
        for t in range(Xt.shape[1]):

            X = Xt[:, t, :, :, :].permute(0, 1, 3, 2)
            Y = Yt[:, t, :, :, :].permute(0, 1, 3, 2)

            if data_range is None:
                data_range = torch.ones_like(Y)  # * Y.max()
                p = (self.win_size - 1) // 2
                data_range = data_range[:, :, p:-p, p:-p]
            data_range = data_range[:, None, None, None]
            C1 = (self.k1 * data_range) ** 2
            C2 = (self.k2 * data_range) ** 2
            ux = F.conv2d(X, self.w)  # typing: ignore
            uy = F.conv2d(Y, self.w)  #
            uxx = F.conv2d(X * X, self.w)
            uyy = F.conv2d(Y * Y, self.w)
            uxy = F.conv2d(X * Y, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux ** 2 + uy ** 2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S = (A1 * A2) / D

            if full:
                ssims += 1 - S
            else:
                ssims += 1 - S.mean()

        return ssims / Xt.shape[1]


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, Xt: torch.Tensor, Yt: torch.Tensor, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)

        ssims = 0.0
        
        Xt = (Xt / (Xt.max()+1e-6)).float().unsqueeze(1)+1e-6
        Yt = (Yt / (Yt.max()+1e-6)).float().unsqueeze(1)+1e-6

        for t in range(Xt.shape[2]):

            X = Xt[:, :, t, :, :]
            Y = Yt[:, :, t, :, :]

            if data_range is None:
                data_range = torch.ones_like(Y)  # * Y.max()
                p = (self.win_size - 1) // 2
                data_range = data_range[:, :, p:-p, p:-p]
            data_range = data_range[:, None, None, None]
            C1 = (self.k1 * data_range) ** 2
            C2 = (self.k2 * data_range) ** 2
            ux = F.conv2d(X, self.w)  # typing: ignore
            uy = F.conv2d(Y, self.w)  #
            uxx = F.conv2d(X * X, self.w)
            uyy = F.conv2d(Y * Y, self.w)
            uxy = F.conv2d(X * Y, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux ** 2 + uy ** 2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S = (A1 * A2) / D

            if full:
                ssims += 1 - S
            else:
                ssims += 1 - S.mean()

        return ssims / Xt.shape[2]


class HighPassLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        high_pass_loss = self.l1_loss(self.high_pass_filter(X), self.high_pass_filter(Y))

        return high_pass_loss

    def high_pass_filter(self, images):
        # Define the high pass filter kernel
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(images.device)

        filtered_images = torch.zeros_like(images)

        # Iterate over the batch, time, and channel dimensions
        for b in range(images.size(0)):
            for t in range(images.size(3)):
                for c in range(images.size(4)):
                    # Apply high pass filter using convolution
                    filtered_image = F.conv2d(images[b, :, :, t, c].unsqueeze(0).unsqueeze(0), kernel, padding=1)
                    filtered_images[b, :, :, t, c] = filtered_image.squeeze()

        return filtered_images


class HighPassImageLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        high_pass_loss = self.l1_loss(self.high_pass_filter(X), self.high_pass_filter(Y))

        return high_pass_loss

    def high_pass_filter(self, images):
        # Define the high pass filter kernel
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(images.device)

        filtered_images = torch.zeros_like(images)

        # Iterate over the batch, time, and channel dimensions
        for b in range(images.size(0)):
            for t in range(images.size(1)):
                # Apply high pass filter using convolution
                filtered_image = F.conv2d(images[b, t, ...].unsqueeze(0), kernel, padding=1)
                filtered_images[b, t, ...] = filtered_image.squeeze()

        return filtered_images


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss


class Eagle_Loss(nn.Module):
    def __init__(self, patch_size=5, cpu=False, cutoff=0.35, epsilon=1e-6):
        super(Eagle_Loss, self).__init__()
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        self.cutoff = cutoff
        self.epsilon = epsilon

        # Scharr kernel for the gradient map calculation
        kernel_values = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        self.kernel_x = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)
        self.kernel_y = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)

        # Operation for unfolding image into non-overlapping patches
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size).to(self.device)

    def forward(self, output, target):
        output, target = output.to(self.device), target.to(self.device)
        if output.size(1) != 1 or target.size(1) != 1:
            raise ValueError("Input 'output' and 'target' should be grayscale")

        # Gradient maps calculation
        gx_output, gy_output = self.calculate_gradient(output)
        gx_target, gy_target = self.calculate_gradient(target)

        # Unfolding and variance calculation
        eagle_loss = self.calculate_patch_loss(gx_output, gx_target) + \
                     self.calculate_patch_loss(gy_output, gy_target)

        return eagle_loss

    def calculate_gradient(self, img):
        img = img.to(self.device)
        gx = F.conv2d(img, self.kernel_x, padding=1, groups=img.shape[1])
        gy = F.conv2d(img, self.kernel_y, padding=1, groups=img.shape[1])
        return gx, gy

    def calculate_patch_loss(self, output_gradient, target_gradient):
        output_gradient, target_gradient = output_gradient.to(self.device), target_gradient.to(self.device)
        batch_size = output_gradient.size(0)
        num_channels = output_gradient.size(1)
        patch_size_squared = self.patch_size * self.patch_size

        # Unfold the gradient tensors into patches
        output_patches = self.unfold(output_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        target_patches = self.unfold(target_gradient).view(batch_size, num_channels, patch_size_squared, -1)

        # Compute the variance for each patch
        var_output = torch.var(output_patches, dim=2, keepdim=True) + self.epsilon
        var_target = torch.var(target_patches, dim=2, keepdim=True) + self.epsilon

        shape0, shape1 = output_gradient.shape[-2] // self.patch_size, output_gradient.shape[-1] // self.patch_size

        # Compute the FFT-based loss on the variance maps
        return self.fft_loss(var_output.view(batch_size, num_channels, shape0, shape1),
                            var_target.view(batch_size, num_channels, shape0, shape1))

    def gaussian_highpass_weights2d(self, size):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = torch.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        return 1 - weights  # Inverted for high pass

    def fft_loss(self, pred, gt):
        pred, gt = pred.to(self.device), gt.to(self.device)

        pred_padded, unpad_pred = self.pad_to_pow2(pred)
        gt_padded, unpad_gt = self.pad_to_pow2(gt)

        pred_fft = torch.fft.fft2(pred_padded)
        gt_fft = torch.fft.fft2(gt_padded)

        # Compute FFT magnitudes
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2 + self.epsilon)
        gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2 + self.epsilon)

        # Normalize FFT magnitudes
        pred_mag = (pred_mag - pred_mag.mean()) / (pred_mag.std() + self.epsilon)
        gt_mag = (gt_mag - gt_mag.mean()) / (gt_mag.std() + self.epsilon)

        # Apply high-pass filter
        # weights = self.gaussian_highpass_weights2d(pred_padded.size()[2:]).to(pred.device)
        weights = self.butterworth_highpass_weights2d(pred_padded.size()[2:], cutoff=self.cutoff, order=2).to(
            pred.device)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        pred_mag_unpadded = unpad_pred(weighted_pred_mag)
        gt_mag_unpadded = unpad_gt(weighted_gt_mag)

        l1_loss_val = F.l1_loss(pred_mag_unpadded, gt_mag_unpadded)
        return l1_loss_val

    def pad_to_pow2(self, x):
        h, w = x.shape[-2:]
        new_h = 1 << (h - 1).bit_length()
        new_w = 1 << (w - 1).bit_length()
        padding = (0, new_w - w, 0, new_h - h)
        padded_x = F.pad(x, padding)

        def unpad(y):
            return y[..., :h, :w]

        return padded_x, unpad

    def butterworth_highpass_weights2d(self, size, cutoff=0.35, order=2):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = 1 / (1 + (cutoff / (freq_mag + self.epsilon)) ** (2 * order))

        return 1 - weights  # Inverted for high-pass filtering

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, layers):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = torch.nn.ModuleList([vgg[i] for i in layers]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0.0
        with torch.no_grad():
            for layer in self.vgg_layers:
                x = layer(x)
                y = layer(y)
                loss += F.mse_loss(x, y)
        return loss
