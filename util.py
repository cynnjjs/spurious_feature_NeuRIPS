import torch
import shutil
import torch.nn.functional as F

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer, source):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    if source:
        test_loss_min = checkpoint['S_test_loss_min']
    else:
        test_loss_min = checkpoint['T_test_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], test_loss_min

def normalize_perturbation(x, scope=None):
    s = x.shape
    x = x.view(s[0], -1)
    x = F.normalize(x, p=2, dim=1)
    x = x.view(s)
    return x

def perturb_image(x, p, model):
    radius = 3.5
    eps = 1e-6 * normalize_perturbation(torch.empty(x.shape).normal_())

    # Predict on randomly perturbed image
    x.requires_grad = True
    eps_p = model(x + eps)
    loss = torch.mean(torch.sum(- F.log_softmax(p, 1) * F.softmax(eps_p, 1), 1))
    loss.backward(retain_graph=True)

    # Based on perturbed image, get direction of greatest error
    eps_adv = x.grad

    # Use that direction as adversarial perturbation
    eps_adv = normalize_perturbation(eps_adv)
    with torch.no_grad():
        x_adv = x + radius * eps_adv

    return x_adv

def vat_loss(x, p, model):
    x_adv = perturb_image(x, p, model)
    p_adv = model(x_adv)
    loss = torch.mean(torch.sum(- F.log_softmax(p, 1) * F.softmax(p_adv, 1), 1))

    return loss
