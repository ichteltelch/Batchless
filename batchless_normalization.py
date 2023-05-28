import tensorflow as tf
from tensorflow.keras.layers import Layer
import math
import numpy as np

class BatchlessNormalization(Layer):
    """
    Normalization/standardization layer that learns the distribution parameters (mean, std) from the input data
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
            Default is 'log'.
        gauge_loss(bool): Whether to shift the loss value so that its expectation is zero if the input samples conform
            to the learned distribution. This does not affect the gradients. Default is True.
        init_mean: initially assumend mean for all channels.
            Default is 0
        init_std: initially assumed standard deviation for all channels.
            Default is 1
    """
    def __init__(
        self,
        shared_axes=None,
        epsilon=1e-5,
        use_output_std=True,
        use_output_mean=True,
        std_parametrization="log",
        gauge_loss=True,
        init_mean = 0.0,
        init_std = 1.0,
        **kwargs
    ):
        super(BatchlessNormalization, self).__init__(**kwargs)
        self.axes = shared_axes
        self.init_mean = init_mean
        self.init_std = init_std
        self.gauge_loss = gauge_loss
        self.epsilon = epsilon
        self.use_output_std = use_output_std
        self.use_output_mean = use_output_mean
        self.std_parametrization = std_parametrization
        #self.specified_input_shape = input_shape
        
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
        if self.axes == None:
            self.axes = list(range(0, len(input_shape)-1))
        else:
            self.axes = np.atleast_1d(self.axes)
            for i, a in enumerate(self.axes):
                a = self.axes[i]
                if a>=len(input_shape):
                    raise f"Specified axis {a} does not exist in input_shape {input_shape}"
                elif a<0:
                    a += len(input_shape)
                    if a<0:
                        raise f"Specified axis {a - len(input_shape)} does not exist in input_shape {input_shape}"
                self.axes[i] = a
            if 0 not in self.axes:
                self.axes = [0] + self.axes
            
        param_shape = [1 if i in self.axes or i == 0 else dim for i, dim in enumerate(input_shape)]
        self.param_shape = param_shape

        self.mean = self.add_weight(shape=param_shape,
                                    initializer=tf.keras.initializers.Constant(value=self.init_mean),
                                    trainable=True,
                                    name='mean',
                                    dtype=self.dtype)

        if self.std_parametrization == 'abs':
            self.std = self.add_weight(shape=param_shape,
                                       initializer=tf.keras.initializers.Constant(value=self.init_std),
                                       trainable=True,
                                       name='std',
                                       dtype=self.dtype)
        elif self.std_parametrization == 'log':
            self.log_std = self.add_weight(shape=param_shape,
                                           initializer=tf.keras.initializers.Constant(value=math.log(self.init_std)),
                                           trainable=True,
                                           name='std',
                                           dtype=self.dtype)
        elif self.std_parametrization == 'inv':
            self.inv_std = self.add_weight(shape=param_shape,
                                           initializer=tf.keras.initializers.Constant(value=1&self.init_std),
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
      Begin accumulating input samples in order to re-initialize the mean.
      Subsequent invocations of call() will record the count and sum of the inputs.
    """
    def begin_observe_mean(self):
        if hasattr(self, 'centered_sample_square_sum'):
            if self.sample_count > 0:
                raise "Cannot start observing samples for mean after samples for variance have already been recorded!"

        self.sample_sum = tf.Variable(shape=self.param_shape,
                                      initial_value = tf.zeros(self.param_shape),
                                      trainable=False,
                                      dtype=self.dtype)
        self.sample_count = 0
    """
      Begin accumulating the squares of the differences between the input samples
      and the currently assumed mean in order to re-initialize the standard deviations.
      Subsequent invocations of call() will record the count and sum of the 
      squares of the inputs after centering them with the currently assumed mean.

      Do not call this method unless the mean has already been computed or is 
      being computed simultanously by means of calling begin_observe_mean() together
      with this method before observing some sample data. 
      Otherwise the result will be wrong!
    """
    def begin_observe_std(self):
        if hasattr(self, 'sample_sum'):
            if self.sample_count > 0:
                raise "Cannot start observing samples for variance after samples for mean have already been recorded!"
        self.centered_sample_square_sum = tf.Variable(shape=self.param_shape,
                                                      initial_value = tf.zeros(self.param_shape),
                                                      trainable=False,
                                                      dtype=self.dtype)
        self.sample_count = 0

    """
    Re-initialize mean and/or standard deviation, depending on whether 
    begin_observe_mean() or begin_observe_mean() have been called.

    You can either 
      - call begin_observe_mean(), call() the layer on some data, then call end_observe(),
        then call begin_observe_std(), call() on the same data, then call end_observe()
        This is more numerically stable
      - or you can call begin_observe_mean() and begin_observe_std(), then 
        call() the layer on some data, then call end_observe().
        This requires only one pass over the data

    Parameters:
      preserve_semantics (bool): if this layer uses output_mean and/or output_std,
        the values in these parametes are changed so that the layer computes the same
        function as before, up to rounding errors. (This feature is untested)
        Default is False

    """
    def end_observe(self, preserve_semantics=False):

        if hasattr(self, 'sample_sum'):
            mean = self.sample_sum / self.sample_count

        if hasattr(self, 'centered_sample_square_sum'):
            var = self.centered_sample_square_sum/self.sample_count
            if hasattr(self, 'sample_sum'):
              # the mean was determined simultaneously. Need to shift values around
              var += 2 * self.mean * mean - tf.square(self.mean) - tf.square(mean)
            std = tf.sqrt(var)

            if preserve_semantics:
                if(self.use_output_std):
                    if hasattr(self, 'std'):
                        old_std = self.std
                    elif hasattr(self, 'log_std'):
                        old_std = tf.exp(self.inv_std)
                    elif hasattr(self, 'inv_std'):
                        old_std = tf.reciprocal(self.inv_std)
                    self.output_std.assign(self.output_std * std / old_std)


            if hasattr(self, 'std'):
                self.std.assign(std)
            elif hasattr(self, 'log_std'):
                self.log_std.assign(tf.math.log(std))
            elif hasattr(self, 'inv_std'):
                self.inv_std.assign(tf.math.reciprocal(std))

            del self.centered_sample_square_sum

        if hasattr(self, 'sample_sum'):

            if preserve_semantics:
                if(self.use_output_mean):
                    if hasattr(self, 'std'):
                        in_std = self.std
                    elif hasattr(self, 'log_std'):
                        in_std = tf.exp(self.inv_std)
                    elif hasattr(self, 'inv_std'):
                        in_std = tf.reciprocal(self.inv_std)
                    self.output_std.assign_add((self.mean-mean)*self.out_std/in_std)


            self.mean.assign(mean)




            del self.sample_sum

        if hasattr(self, 'sample_count'):
            del self.sample_count
        
    
    def call(self, inputs, training=None):
        
        if hasattr(self, 'sample_count'):
            if training:
                raise "Cannot train BlN layer: still accumulating input statistics"
            self.sample_count += 1
        
        if hasattr(self, 'sample_sum'):
            self.sample_sum.assign_add(tf.reduce_mean(inputs, axis=self.axes, keepdims=True))
        
        centered_inputs_for_output = (inputs - tf.stop_gradient(self.mean))
        
        if hasattr(self, 'centered_sample_square_sum'):
            self.centered_sample_square_sum.assign_add(
                tf.reduce_mean(tf.square(centered_inputs_for_output), 
                               axis=self.axes, keepdims=True))


        gauge_loss = self.gauge_loss
        compute_inference_loss = True
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
           

        

        normalized_inputs_for_output = centered_inputs_for_output * tf.stop_gradient(inv_std)

        if self.use_output_std and self.output_std is not None:
            outputs = normalized_inputs_for_output * self.output_std
        else:
            outputs = normalized_inputs_for_output

        if self.use_output_mean and self.output_mean is not None:
            outputs += self.output_mean

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

        return outputs

