import torch


def cce(p, y):
    eps = 1e-10
    loss_ = y * torch.log(p + eps)
    return -torch.mean(torch.sum(loss_, 1))


class JSD(torch.nn.Module):

    def __init__(self, alpha=0.5):
        super(JSD, self).__init__()
        self.alpha = alpha

    def forward(self, p, y):
        y = torch.nn.functional.one_hot(y.to(torch.int64), p.shape[-1]).long()
        p = torch.nn.Softmax(dim=1)(p)

        beta = 1 - self.alpha
        alpha = self.alpha

        m = alpha * y + beta * p
        loss_a = alpha * cce(p, m)
        loss_b = beta * cce(y, m)
        return loss_a + loss_b
