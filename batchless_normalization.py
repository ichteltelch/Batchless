import tensorflow as tf
from tensorflow.keras.layers import Layer
import math


class BatchlessNormalization(Layer):
    """
    Custom normalization/standardization layer that learns the distribution parameters (mean, std) from the input data
    using gradient descent on the negative logarithmic likelihood of a Gaussian with these parameters. 
    and uses them to normalize the input. It also supports optional scaling and shifting of the normalized input.

    Parameters:
        axes (list): List of axes along which the mean and standard deviation parameters are shared. Default is None,
            which means only the batch axis is shared. The batch axis is always shared.
        epsilon (float): Small constant added to denominators and logarithm-function arguments for numerical stability. Default is 1e-5.
        use_output_std (bool): Whether to learn output standard deviation parameters for scaling the normalized input.
            Default is True.
        use_output_mean (bool): Whether to learn output mean parameters for shifting the normalized input. Default is True.
        std_parametrization (str): Parameterization for the standard deviation. Possible values are 'abs', 'log', or 'inv'.
            Default is 'abs'.
            gauge_loss(bool): Whether to shift the loss value so that its expectation is zero if the input samples conform
            to the learned distribution. This does not affect the gradients. Default is True.
    """
    def __init__(
        self,
        axes=None,
        epsilon=1e-5,
        use_output_std=True,
        use_output_mean=True,
        std_parametrization="abs",
        gauge_loss=True,
        **kwargs
    ):
        super(BatchlessNormalization, self).__init__(**kwargs)
        if axes is None:
            axes = []  # Default to no specific axes
        self.axes = axes
        self.epsilon = epsilon
        self.use_output_std = use_output_std
        self.use_output_mean = use_output_mean
        self.std_parametrization = std_parametrization
        
    def get_config(self):
        config = super(BatchlessNormalization, self).get_config()
        config.update({
            "axes":self.axes.tolist(),
            "epsilon":self.epsilon,
            "use_output_std":self.use_output_std,
            "use_output_mean":self.use_output_mean,
            "std_parametrization":self.std_parametrization
        })
        return config

    def build(self, input_shape):
        param_shape = [1 if i in self.axes or i == 0 else dim for i, dim in enumerate(input_shape)]

        self.mean = self.add_weight(shape=param_shape,
                                    initializer='zeros',
                                    trainable=True,
                                    name='mean',
                                    dtype=self.dtype)

        if self.std_parametrization == 'abs':
            self.std = self.add_weight(shape=param_shape,
                                       initializer='ones',
                                       trainable=True,
                                       name='std',
                                       dtype=self.dtype)
        elif self.std_parametrization == 'log':
            self.log_std = self.add_weight(shape=param_shape,
                                           initializer='zeros',
                                           trainable=True,
                                           name='std',
                                           dtype=self.dtype)
        elif self.std_parametrization == 'inv':
            self.inv_std = self.add_weight(shape=param_shape,
                                           initializer='ones',
                                           trainable=True,
                                           name='std',
                                           dtype=self.dtype)
        else:
            raise ValueError("Invalid std_parametrization value. Possible values are 'abs', 'log', or 'inv'.")

        if self.use_output_mean:
            self.output_mean = self.add_weight(shape=param_shape,
                                               initializer='zeros',
                                               trainable=True,
                                               name='output_mean',
                                               dtype=self.dtype)
        else:
            self.output_mean = None

        if self.use_output_std:
            self.output_std = self.add_weight(shape=param_shape,
                                              initializer='ones',
                                              trainable=True,
                                              name='output_std',
                                              dtype=self.dtype)
        else:
            self.output_std = None
    """
    Parameters:
        inputs (tensor): The input activations
        training (bool): Whether the layer is training
        compute_inference_loss (bool): whether to compute the loss at inverence time at all
    """
    def call(self, inputs, training=None, compute_inference_loss=False):
        gauge_loss = self.gauge_loss

        inv_std = None
        log_std = None
        needs_log = (training or compute_inference_loss and not gauge_loss)

        if hasattr(self, 'std'):
            inv_std = tf.math.reciprocal(tf.abs(self.std) + self.epsilon)
            if needs_log: 
                log_std = tf.math.log(tf.abs(self.std) + self.epsilon)
        elif hasattr(self, 'log_std'):
            inv_std = tf.math.exp(-self.log_std)
            if needs_log: 
                log_std = self.log_std
        elif hasattr(self, 'inv_std'):
            inv_std = self.inv_std
            if needs_log: 
                log_std = -tf.math.log(tf.abs(self.inv_std) + self.epsilon)

        normalized_inputs_for_output = (inputs - tf.stop_gradient(self.mean)) * tf.stop_gradient(inv_std)

        if self.use_output_std and self.output_std is not None:
            scaled_inputs = normalized_inputs_for_output * self.output_std
        else:
            scaled_inputs = normalized_inputs_for_output

        if self.use_output_mean and self.output_mean is not None:
            scaled_inputs += self.output_mean

        if training or compute_inference_loss:
            
            if training:
                normalized_inputs_for_loss = (tf.stop_gradient(inputs) - self.mean) * inv_std
            else:
                normalized_inputs_for_loss = normalized_inputs_for_output

            # Calculate and add the custom loss
            # Note that we omit the term log(2π)/2 which theoretically also arises.
            if training or not gauge_loss:
                mean_log = tf.reduce_mean(log_std)
                loss = mean_log + 0.5 * tf.reduce_mean(tf.square(normalized_inputs_for_loss))
                if gauge_loss:
                    expected_loss = mean_log
                    loss += - tf.stop_gradient(expected_loss) - 0.5
            else:
                # when we don't need gradients and the expected loss is gauged towards zero
                # we don't actually need the logarithms of the standard deviations
                loss = 0.5 * (tf.reduce_mean(tf.square(normalized_inputs_for_loss)) - 1)        
        
        self.add_loss(loss)

        return scaled_inputs