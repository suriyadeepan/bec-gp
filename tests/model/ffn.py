import pytest
import torch


@pytest.fixture
def sregressor():
  from neuralbec.model.ffn import SimpleRegressor
  return SimpleRegressor(1, 512)


def test_SimpleRegressor(sregressor):
  low, high = 0., 500.
  output = sregressor(torch.FloatTensor(1, 1).uniform_(low, high))
  assert output.size() == (1, 512)
