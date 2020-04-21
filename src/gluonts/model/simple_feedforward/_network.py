# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import Tuple, List

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.block.scaler import MeanScaler, NOPScaler
from gluonts.core.component import validated
from gluonts.distribution import Distribution, DistributionOutput
from gluonts.model.common import Tensor
from gluonts.representation import Representation


class SimpleFeedForwardNetworkBase(mx.gluon.HybridBlock):
    """
    Abstract base class to implement feed-forward networks for probabilistic
    time series prediction.

    This class does not implement hybrid_forward: this is delegated
    to the two subclasses SimpleFeedForwardTrainingNetwork and
    SimpleFeedForwardPredictionNetwork, that define respectively how to
    compute the loss and how to generate predictions.

    Parameters
    ----------
    num_hidden_dimensions
        Number of hidden nodes in each layer.
    prediction_length
        Number of time units to predict.
    context_length
        Number of time units that condition the predictions.
    batch_normalization
        Whether to use batch normalization.
    distr_output
        Distribution to fit.
    kwargs
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        num_hidden_dimensions: List[int],
        prediction_length: int,
        context_length: int,
        batch_normalization: bool,
        input_repr: Representation,
        output_repr: Representation,
        distr_output: DistributionOutput,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.input_repr = input_repr
        self.output_repr = output_repr
        self.distr_output = distr_output

        with self.name_scope():
            self.distr_args_proj = self.distr_output.get_args_proj()
            self.mlp = mx.gluon.nn.HybridSequential()
            dims = self.num_hidden_dimensions
            for layer_no, units in enumerate(dims[:-1]):
                self.mlp.add(mx.gluon.nn.Dense(units=units, activation="relu"))
                if self.batch_normalization:
                    self.mlp.add(mx.gluon.nn.BatchNorm())
            self.mlp.add(mx.gluon.nn.Dense(units=prediction_length * dims[-1]))
            self.mlp.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(
                        o, (-1, prediction_length, dims[-1])
                    )
                )
            )

    def get_distr(self, F, past_target: Tensor) -> Tuple[Distribution, Tensor]:
        """
        Given past target values, applies the feed-forward network and
        maps the output to a probability distribution for future observations.

        Parameters
        ----------
        F
        past_target
            Tensor containing past target observations.
            Shape: (batch_size, context_length, target_dim).

        Returns
        -------
        Distribution
            The predicted probability distribution for future observations.
        """

        input_tar_repr, scale = self.input_repr(
            past_target, F.ones_like(past_target), None
        )

        self.output_repr(past_target, F.ones_like(past_target), None)

        mlp_outputs = self.mlp(input_tar_repr)

        distr_args = self.distr_args_proj(mlp_outputs)
        return self.distr_output.distribution(distr_args, scale=scale), scale


class SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, past_target: Tensor, future_target: Tensor
    ) -> Tensor:
        """
        Computes a probability distribution for future data given the past,
        and returns the loss associated with the actual future observations.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        future_target
            Tensor with future observations.
            Shape: (batch_size, prediction_length, target_dim).

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size, ).
        """
        distr, _ = self.get_distr(F, past_target)

        output_tar_repr, _ = self.output_repr(
            future_target, F.ones_like(future_target), None
        )

        # (batch_size, prediction_length, target_dim)
        loss = distr.loss(output_tar_repr)

        # (batch_size, )
        return loss.mean(axis=1)


class SimpleFeedForwardPredictionNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, past_target: Tensor) -> Tensor:
        """
        Computes a probability distribution for future data given the past,
        and draws samples from it.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).

        Returns
        -------
        Tensor
            Prediction sample. Shape: (batch_size, samples, prediction_length).
        """

        distr, scale = self.get_distr(F, past_target)

        self.output_repr(past_target, F.ones_like(past_target), scale)

        # (num_samples, batch_size, prediction_length)
        samples = distr.sample(self.num_parallel_samples)

        samples = samples.swapaxes(1, 2)
        tranf_all_samples = []

        for i in range(self.num_parallel_samples):
            tranf_loc_samples = []
            for j in range(self.prediction_length):
                samples_sub = F.slice_axis(
                    samples, begin=i, end=i + 1, axis=0
                ).squeeze(axis=0)
                samples_sub = F.slice_axis(
                    samples_sub, begin=j, end=j + 1, axis=0
                ).squeeze(axis=0)
                samples_sub = F.squeeze(
                    self.output_repr.post_transform(F, samples_sub)
                )
                tranf_loc_samples.append(samples_sub.expand_dims(axis=-1))
            tranf_all_samples.append(
                F.concat(*tranf_loc_samples, dim=-1).expand_dims(axis=-1)
            )

        # (batch_size, prediction_length, num_samples)
        samples = F.concat(*tranf_all_samples, dim=-1)

        # (batch_size, num_samples, prediction_length)
        return samples.swapaxes(1, 2)
