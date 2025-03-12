# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalization layers."""
import torch.nn as nn
import torch
import functools


def get_normalization(config, conditional=False):
  """Obtain normalization modules from the config file."""
  norm = config.model.normalization
  if conditional:
    if norm == 'InstanceNorm++':
      return functools.partial(ConditionalInstanceNorm2dPlus, num_classes=config.model.num_classes)
    else:
      raise NotImplementedError(f'{norm} not implemented yet.')
  else:
    if norm == 'InstanceNorm':
      return nn.InstanceNorm2d
    elif norm == 'InstanceNorm++':
      return InstanceNorm2dPlus
    elif norm == 'VarianceNorm':
      return VarianceNorm2d
    elif norm == 'GroupNorm':
      return nn.GroupNorm
    else:
      raise ValueError('Unknown normalization: %s' % norm)


class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    if self.bias:
      self.embed = nn.Embedding(num_classes, num_features * 2)
      self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
      self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
    else:
      self.embed = nn.Embedding(num_classes, num_features)
      self.embed.weight.data.uniform_()

  def forward(self, x, y):
    out = self.bn(x)
    if self.bias:
      gamma, beta = self.embed(y).chunk(2, dim=1)
      out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    else:
      gamma = self.embed(y)
      out = gamma.view(-1, self.num_features, 1, 1) * out
    return out


class ConditionalInstanceNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    if bias:
      self.embed = nn.Embedding(num_classes, num_features * 2)
      self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
      self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
    else:
      self.embed = nn.Embedding(num_classes, num_features)
      self.embed.weight.data.uniform_()

  def forward(self, x, y):
    h = self.instance_norm(x)
    if self.bias:
      gamma, beta = self.embed(y).chunk(2, dim=-1)
      out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
    else:
      gamma = self.embed(y)
      out = gamma.view(-1, self.num_features, 1, 1) * h
    return out


class ConditionalVarianceNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, bias=False):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.embed = nn.Embedding(num_classes, num_features)
    self.embed.weight.data.normal_(1, 0.02)

  def forward(self, x, y):
    vars = torch.var(x, dim=(2, 3), keepdim=True)
    h = x / torch.sqrt(vars + 1e-5)

    gamma = self.embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * h
    return out


class VarianceNorm2d(nn.Module):
  def __init__(self, num_features, bias=False):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.alpha = nn.Parameter(torch.zeros(num_features))
    self.alpha.data.normal_(1, 0.02)

  def forward(self, x):
    vars = torch.var(x, dim=(2, 3), keepdim=True)
    h = x / torch.sqrt(vars + 1e-5)

    out = self.alpha.view(-1, self.num_features, 1, 1) * h
    return out


class ConditionalNoneNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    if bias:
      self.embed = nn.Embedding(num_classes, num_features * 2)
      self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
      self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
    else:
      self.embed = nn.Embedding(num_classes, num_features)
      self.embed.weight.data.uniform_()

  def forward(self, x, y):
    if self.bias:
      gamma, beta = self.embed(y).chunk(2, dim=-1)
      out = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)
    else:
      gamma = self.embed(y)
      out = gamma.view(-1, self.num_features, 1, 1) * x
    return out


class NoneNorm2d(nn.Module):
  def __init__(self, num_features, bias=True):
    super().__init__()

  def forward(self, x):
    return x


class InstanceNorm2dPlus(nn.Module):
  def __init__(self, num_features, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    self.alpha = nn.Parameter(torch.zeros(num_features))
    self.gamma = nn.Parameter(torch.zeros(num_features))
    self.alpha.data.normal_(1, 0.02)
    self.gamma.data.normal_(1, 0.02)
    if bias:
      self.beta = nn.Parameter(torch.zeros(num_features))

  def forward(self, x):
    means = torch.mean(x, dim=(2, 3))
    m = torch.mean(means, dim=-1, keepdim=True)
    v = torch.var(means, dim=-1, keepdim=True)
    means = (means - m) / (torch.sqrt(v + 1e-5))
    h = self.instance_norm(x)

    if self.bias:
      h = h + means[..., None, None] * self.alpha[..., None, None]
      out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
    else:
      h = h + means[..., None, None] * self.alpha[..., None, None]
      out = self.gamma.view(-1, self.num_features, 1, 1) * h
    return out


class ConditionalInstanceNorm2dPlus(nn.Module):
  def __init__(self, num_features, num_classes, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    if bias:
      self.embed = nn.Embedding(num_classes, num_features * 3)
      self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
      self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
    else:
      self.embed = nn.Embedding(num_classes, 2 * num_features)
      self.embed.weight.data.normal_(1, 0.02)

  def forward(self, x, y):
    means = torch.mean(x, dim=(2, 3))
    m = torch.mean(means, dim=-1, keepdim=True)
    v = torch.var(means, dim=-1, keepdim=True)
    means = (means - m) / (torch.sqrt(v + 1e-5))
    h = self.instance_norm(x)

    if self.bias:
      gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
      h = h + means[..., None, None] * alpha[..., None, None]
      out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
    else:
      gamma, alpha = self.embed(y).chunk(2, dim=-1)
      h = h + means[..., None, None] * alpha[..., None, None]
      out = gamma.view(-1, self.num_features, 1, 1) * h
    return out
