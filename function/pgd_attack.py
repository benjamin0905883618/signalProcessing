import torch
import torch.nn.functional as F


class attack(object):
    def __init__(self, model, mode = 'FGSM'):
        self.model = model
        if mode == 'FGSM':
            self.epsilon = 8/255
            self.alpha = 2/255
            self.iter = 40
        elif 'BIM':
            self.epsilon = 8/255
            self.alpha = 2/255
            self.iter = 10
        elif 'FGSM':
            self.epsilon = 8/255
            self.alpha = 8/255
            self.iter = 1
            
    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.iter):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x, x - x_natural
    
