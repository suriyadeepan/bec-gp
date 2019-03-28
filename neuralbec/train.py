import torch
from neuralbec import utils
from tqdm import tqdm
import numpy as np

logger = utils.get_logger(__name__)
# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit(model, dataset, hparams, epochs):
  trainset, testset = dataset
  X, psi = trainset
  batch_size = hparams['batch_size']
  loss_fn = hparams['loss_fn']
  optim = hparams['optim'](model.parameters())
  iterations = len(X) // batch_size
  # iterations = 50

  for epoch in range(epochs):
    # put model in train mode
    model.train()
    # keep track of steps; `i` is not to be trusted
    steps = 0
    epoch_loss = 0.
    for i in tqdm(range(iterations)):
      # clear grad
      optim.zero_grad()
      # get batches
      batch_x = X[i * batch_size : (i + 1) * batch_size ]
      batch_psi = psi[i * batch_size : (i + 1) * batch_size ]

      # make sure batch size checks out
      if not len(batch_x) == batch_size:
        continue

      batch_x = torch.tensor(batch_x).view(batch_size, -1)
      batch_psi = torch.tensor(batch_psi).view(batch_size, -1)

      assert batch_x.size() == (batch_size, 1), batch_x.size()
      assert batch_psi.size() == (batch_size, 512), batch_psi.size()

      # forward
      psi_hat = model(batch_x.float())
      # loss
      loss = loss_fn(batch_psi.double(), psi_hat.double())
      # calculate gradients
      loss.backward()
      # update parameters
      optim.step()
      # calculate steps
      steps += 1
      epoch_loss += loss.item()
    logger.info('[{}] Epoch Loss : {}'.format(epoch, epoch_loss / steps))
    # evaluate
    eval_loss, accuracy = evaluate(model, testset, hparams)
    logger.info('[Evaluation] Loss : {}, Accuracy : {}'.format(eval_loss, accuracy))


def evaluate(model, testset, hparams):
  X, psi = testset
  batch_size = hparams['batch_size']
  loss_fn = hparams['loss_fn']
  iterations = len(X) // batch_size

  # put model in eval mode
  model.eval()
  # track losses
  losses = []
  for i in tqdm(range(iterations)):
    # get batches
    batch_x = X[i * batch_size : (i + 1) * batch_size ]
    batch_psi = psi[i * batch_size : (i + 1) * batch_size ]

    # make sure batch size checks out
    if not len(batch_x) == batch_size:
      continue

    # np.array's to torch.tensor's
    batch_x = torch.tensor(batch_x).view(batch_size, -1)
    batch_psi = torch.tensor(batch_psi).view(batch_size, -1)

    # forward
    psi_hat = model(batch_x.float())
    # loss
    losses.append(loss_fn(batch_psi.double(), psi_hat.double()).item())

  loss = np.array(losses).mean()

  return loss, 0
