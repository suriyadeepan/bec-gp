import torch
from neuralbec import utils
from tqdm import tqdm

logger = utils.get_logger(__name__)
# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, dataset, hparams, epochs):
  trainset, testset = dataset
  X, psi = trainset
  batch_size = hparams['batch_size']
  loss_fn = hparams['loss_fn']
  optim = hparams['optim']
  # iterations = len(X) // batch_size
  iterations = 50

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
      batch_x = torch.tensor(batch_x).view(batch_size, -1)
      batch_psi = torch.tensor(batch_psi).view(batch_size, 512)
      logger.debug(batch_x.size())
      logger.debug(batch_psi.size())
      # forward
      psi_hat = model(batch_x)
      # loss
      loss = loss_fn(batch_psi.double(), psi_hat.double())
      # calculate gradients
      loss.backward()
      # update parameters
      optim.step()
      # calculate steps
      steps += 1
      epoch_loss += loss.item()
    logger.info(epoch_loss / steps)
