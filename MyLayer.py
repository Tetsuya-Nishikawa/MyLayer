#tensorflow(version2)で使用する自作Layerモジュール
#論文：https://arxiv.org/pdf/1801.07455.pdfの(9)式を実装
#self.Mは、(10)式以下に記載されている重みのこと
class GcnLayer(tf.keras.layers.Layer):
    def __init__(self, input_dimention,output_dimention, nodes, Pooling=True):
        super(GcnLayer, self).__init__()
        self.output_dimention = output_dimention #出力次元
        self.input_dimention = input_dimention      #入力次元
        if Pooling==False:
            self.adjacency_matrix,self.degree_matrix  = tf.cast(construct_graph.GetAdjacencyAndDegree_Matrix('graph.txt'), tf.float32) #隣接行列と次数行列
        else:
            self.adjacency_matrix,self.degree_matrix  = tf.cast(construct_graph.GetAdjacencyAndDegree_Matrix('graph_after_pooling.txt'), tf.float32) #隣接行列と次数行列  
        
        self.nodes = nodes #ノード数

    def build(self, input_shape):
        self.w = self.add_weight(shape=[self.input_dimention, self.output_dimention], initializer = tf.keras.initializers.he_normal(), trainable=True)
        self.M = tf.Variable(initial_value=tf.ones((self.nodes, self.nodes)), trainable=True)
        
    #https://www.tensorflow.org/guide/keras/masking_and_padding　の　”Supporting masking in your custom layers”　にcompute_maskの解説あり
    def compute_mask(self, inputs, mask=None):
        return mask
    
    #https://www.tensorflow.org/guide/keras/masking_and_padding　の　”Writing layers that need mask information”　に引数にmaskを指定する解説
    def call(self, inputs, mask=None):
        
        A_ElementProduct_M =  tf.multiply(self.adjacency_matrix,self.M)
        L = tf.matmul(self.degree_matrix, tf.matmul(A_ElementProduct_M, self.degree_matrix))
        outputs = tf.matmul(L, tf.matmul(inputs,self.w))
        return tf.nn.relu(outputs)

    
 #LSTMCell
#https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/rnn_cell_impl.py#L804-L1081を参考にしました。
class GcnLstmCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, units, name, Pooling,**kwargs):
   
        super(GcnLstmCell, self).__init__(**kwargs)
        self.nodes = units[0]
        state_size  = tf.TensorShape(units)
        self._state_size = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_size, state_size)
        self._output_size = tf.TensorShape(state_size)
        #self.Iとself.D_selfは、ここでは使用しません…
        self.A , self.I , self.D, self.D_self = construct_graph.GetGraphMatrix() 
        #グラフプーリング使用したかどうかで、使用するグラフを決める
        if Pooling==False:
            self.A, self.D = construct_graph.GetAdjacencyAndDegree_Matrix('graph.txt')
        else:
            self.A, self.D = construct_graph.GetAdjacencyAndDegree_Matrix('graph_after_pooling.txt')
        self.input_channels = units[1]
        self.output_channels = units[1]
        self.cell_name = name
        
        self.batch_xi = tf.keras.layers.BatchNormalization()
        self.batch_hi = tf.keras.layers.BatchNormalization()

        self.batch_xg = tf.keras.layers.BatchNormalization()
        self.batch_hg = tf.keras.layers.BatchNormalization()
        

        self.batch_xf = tf.keras.layers.BatchNormalization()
        self.batch_hf = tf.keras.layers.BatchNormalization()

        self.batch_xo = tf.keras.layers.BatchNormalization()
        self.batch_ho = tf.keras.layers.BatchNormalization()
        self.batch_cell = tf.keras.layers.BatchNormalization()
        
    def build(self, input_shape):

        self.Wxi = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="wxi"+self.cell_name)
        self.Whi = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="whi"+self.cell_name)
        self.Wxg = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="wxg"+self.cell_name)
        self.Whg = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="whg"+self.cell_name)
        self.Wxf = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="wxf"+self.cell_name)
        self.Whf = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="wfg"+self.cell_name)
        self.Wxo = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="wxo"+self.cell_name)
        self.Who = self.add_weight(shape=[self.input_channels, self.output_channels], initializer = tf.keras.initializers.he_normal(), trainable=True, name="who"+self.cell_name)
        self.M = self.add_weight(shape=[self.nodes, self.nodes], initializer=tf.ones_initializer(), trainable=True,  name="M"+self.cell_name)
        self.built = True

    @property
    def output_size(self):
        return self._output_size
    @property
    def state_size(self):
        return self._state_size
    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask
    def call(self, inputs, states, mask=None, training=True):
        cell, hidden = states
        DAD = tf.matmul(self.D, tf.matmul(self.A*self.M, self.D))
        #f
        output_xf    = tf.matmul(DAD, tf.matmul(inputs, self.Wxf))
        forget_gate = tf.matmul(DAD, tf.matmul(hidden, self.Whf))
        output_xf    =  self.batch_xf(output_xf, training=training)
        forget_gate =  self.batch_hf(forget_gate, training=training)

        forget_gate = output_xf + forget_gate
        forget_gate = tf.nn.sigmoid(forget_gate)

        #i
        output_xi    = tf.matmul(DAD, tf.matmul(inputs, self.Wxi))
        input_gate =  tf.matmul(DAD, tf.matmul(hidden, self.Whi))
        output_xi    = self.batch_xi(output_xi, training=training)
        input_gate =  self.batch_hi(input_gate, training=training)

        input_gate = output_xi + input_gate
        input_gate = tf.nn.sigmoid(input_gate)

        #g
        #tensorflowの仕様では、newinputと呼んでるのでここでもそれに従う
        output_xg    = tf.matmul(DAD, tf.matmul(inputs, self.Wxg))
        newinput_gate =  tf.matmul(DAD, tf.matmul(hidden, self.Whg))
        output_xg    = self.batch_xg(output_xg, training=training)
        newinput_gate =  self.batch_hg(newinput_gate, training=training)

        newinput_gate = output_xg + newinput_gate
        newinput_gate = tf.nn.tanh(newinput_gate)

        #o
        output_xo    = tf.matmul(DAD, tf.matmul(inputs, self.Wxo))
        output_gate = tf.matmul(DAD, tf.matmul(hidden, self.Who))
        output_xo = self.batch_xo(output_xo, training=training)
        output_gate =self.batch_ho(output_gate, training=training)

        output_gate = output_xo + output_gate
        output_gate = tf.nn.sigmoid(output_gate)

        #statesの更新
        new_cell      = forget_gate * cell + (input_gate * newinput_gate)
        new_cell      = self.batch_cell(new_cell, training=training)
        
        new_hidden = output_gate * tf.nn.tanh(new_cell)
        new_state   = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_cell, new_hidden)
        
        return new_hidden, new_state
  
  
class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filter_shape, input_channels, filters, strides_shape, padding):
        super(ConvLayer, self).__init__()
        self.conv_filt_shape = [filter_shape[0], filter_shape[1],  input_channels, filters]
        self.filters = filters
        self.strides             = strides_shape
        self.padding = padding
    def build(self, input_shape):
        self.w = self.add_weight(shape=self.conv_filt_shape, initializer = tf.keras.initializers.he_normal(), trainable=True)
        self.b = self.add_weight(shape=[self.filters,], initializer = tf.keras.initializers.he_normal(), trainable=True)
    
    def compute_mask(self, inputs, mask=None):
        return mask
 
    def call(self, inputs, mask=None):
        filter_shape = [self.conv_filt_shape[0], self.conv_filt_shape[1]]
        #ストライドが2の時は、元の画像サイズに対しての半分に調整
        if self.strides[0] == 2:
            pad = -(-(-2+filter_shape[0])//2)
            inputs = MyLibrary.zero_padding(inputs, pad)
        #inputs   = batch_layer(inputs, 3)
        
        outputs = tf.nn.conv2d(inputs, self.w, strides=self.strides, padding=self.padding) + self.b
        return tf.nn.relu(outputs)

class ResNetLayer(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(ResNetLayer, self).__init__()
        self.filters = filters
        self.strides = strides
        
    def build(self, input_shape):
        input_dimention = input_shape[3]
        if self.strides[0]==2:
            self.conv1                = ConvLayer([1, 1], input_dimention, self.filters//4, self.strides, "VALID")
        else:
            self.conv1                = ConvLayer([1, 1], input_dimention, self.filters//4, self.strides, "SAME")          
        self.conv2                = ConvLayer([3, 3], self.filters//4,                 self.filters//4, [1, 1], "SAME")
        self.conv3                = ConvLayer([1, 1], self.filters//4,                 self.filters     , [1, 1], "SAME")
        if self.strides[0]==2:
            self.shortcut = ConvLayer([3, 3], input_dimention, self.filters, [2, 2], "VALID")
        else:
            self.shortcut = ConvLayer([3, 3], input_dimention, self.filters, [1, 1], "SAME")


    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):

        block1 = self.conv1(inputs)
        block2 = self.conv2(block1)
        outputs = self.conv3(block2)
      
        outputs = outputs + self.shortcut(inputs)

        return tf.nn.relu(outputs)

    
    
#ConvLSTM(https://arxiv.org/pdf/1506.04214v1.pdf)
class ConvLSTM2D(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, units, filters, **kwargs):
        super(ConvLSTM2D, self).__init__(**kwargs)
        self.filters = filters
        #units->(height, width, channels)
        units[0] = units[0] - 3 + 1
        units[1] = units[1] - 3 + 1
        units[2] = self.filters
        state_size  = tf.TensorShape(units)
        self._state_size = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_size, state_size)
        self._output_size = tf.TensorShape(state_size)

    def build(self, input_shape):
        
        #he_normal(https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)
        #tf.keras.layers.Conv2Dに対して、activationを指定しない場合、活性化関数を使用しないことになる。
        self.conv_xi = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_hi = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        self.conv_xf = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_hf = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        self.conv_xo = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_ho = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        self.conv_xg = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal")
        self.conv_hg = tf.keras.layers.Conv2D(self.filters, 3, kernel_initializer="he_normal", padding="same")
        #RECURRENT BATCH NORMALIZATION(https://arxiv.org/pdf/1603.09025.pdf)
        self.batch_xi = tf.keras.layers.BatchNormalization()
        self.batch_hi = tf.keras.layers.BatchNormalization()
        self.batch_xf = tf.keras.layers.BatchNormalization()
        self.batch_hf = tf.keras.layers.BatchNormalization()
        self.batch_xo = tf.keras.layers.BatchNormalization()
        self.batch_ho = tf.keras.layers.BatchNormalization()
        self.batch_xg = tf.keras.layers.BatchNormalization()
        self.batch_hg = tf.keras.layers.BatchNormalization()
        self.batch_cell = tf.keras.layers.BatchNormalization()
        self.build = True
    
    @property
    def output_size(self):
        return self._output_size
    @property
    def state_size(self):
        return self._state_size   

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask
    
    def call(self, inputs, states, mask=None, training=True):
        cell, hidden = states
        f = tf.nn.sigmoid(self.batch_xf(self.conv_xf(inputs)) + self.batch_hf(self.conv_hf(hidden)))
        i = tf.nn.sigmoid(self.batch_xi(self.conv_xi(inputs)) + self.batch_hi(self.conv_hi(hidden)))
        o = tf.nn.sigmoid(self.batch_xo(self.conv_xo(inputs)) + self.batch_hi(self.conv_ho(hidden)))
        g = tf.nn.tanh(self.batch_xg(self.conv_xg(inputs)) + self.batch_hg(self.conv_hg(hidden)))
        #statesの更新!!       
        new_cell      = f * cell + (i * g)
        new_cell      = self.batch_cell(new_cell, training=training)
        
        new_hidden = o * tf.nn.tanh(new_cell)
        new_state   = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_cell, new_hidden)
        
        return new_hidden, new_state
