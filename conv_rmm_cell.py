import tensorflow as tf

class ConvCell(tf.nn.rnn_cell.RNNCell): 
    def __init__(self, input_shape, filters, kernel_size, stride=1, padding='VALID', activation=None, reuse=None, name=None):
        super(ConvCell, self).__init__(_reuse=reuse, name=name)
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._activation = activation
        self._input_shape = input_shape
        
        self.build()

    def build(self):
        self._kernel = self.add_variable('conv_kernel', 
                                         shape=[self._kernel_size, self._input_shape[-1], self._filters],
                                         initializer=tf.contrib.layers.xavier_initializer())
               
        self._num_units = (self._input_shape[1] - self._kernel_size) // self._stride + 1 if self._padding.upper() == 'VALID' else self._input_shape[1]
        
        #self._recurrent_kernel = self.add_variable('recurent_kernel',
        #                                           shape=[1, self._num_units, self._filters], 
        #                                           initializer=tf.contrib.layers.xavier_initializer())
               
        #self._bias = self.add_variable('bias',
        #                               shape=[1, self._num_units, self._filters], 
        #                               initializer=tf.contrib.layers.xavier_initializer())
        
        self._recurrent_kernel = self.add_variable('recurent_kernel',
                                                   shape=[1, 1, self._filters], 
                                                   initializer=tf.contrib.layers.xavier_initializer())
               
        self._bias = self.add_variable('bias',
                                       shape=[1, 1, self._filters], 
                                       initializer=tf.contrib.layers.xavier_initializer())
        
        self.built = True
               

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "ConvCell"
            conv = tf.nn.conv1d(inputs, 
                                filters=self._kernel,
                                stride=self._stride,
                                padding=self._padding.upper(),
                                name='conv1d')
               
            
            recurrent = self._recurrent_kernel * state
               
            outputs =  conv + recurrent + self._bias
               
            if self._activation:
                outputs = self._activation(outputs)

        return outputs, outputs
    
    @property
    def output_size(self):
        return tf.TensorShape([self._num_units, self._filters])
    
    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return tf.TensorShape([self._num_units, self._filters])
