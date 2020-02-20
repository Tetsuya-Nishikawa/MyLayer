#tensorflow(version2)で使用する自作Layerモジュール
#論文：https://arxiv.org/pdf/1801.07455.pdfの(9)式を実装
#self.Mは、(10)式以下に記載されている重みのこと

class GcnLayer(tf.keras.layers.Layer):
    def __init__(self, input_dimention,output_dimention):
        super(GcnLayer, self).__init__()
        self.output_dimention = output_dimention #出力次元
        self.input_dimention = input_dimention      #入力次元
        #construct_graph.MakeAdjacencyMatrix()は、自作関数で隣接行列と次数行列を返す関数
        self.adjacency_matrix,self.degree_matrix  = tf.cast(tf.convert_to_tensor(construct_graph.MakeAdjacencyMatrix()), tf.float32) #隣接行列と次数行列
        self.nodes = 49 #ノード数(今使用しているグラフのノード数が49)

    def build(self, input_shape):
        
        #self.wの初期化方法はHeの初期値(https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)
        #活性化関数Reluを使用する場合、Heの初期値が有効らしい
        self.w = self.add_weight(shape=[self.input_dimention, self.output_dimention], initializer = tf.keras.initializers.he_normal(), trainable=True)
        self.M = tf.Variable(initial_value=tf.ones((self.nodes, self.nodes)), trainable=True)
        
    #とりあえず、compute_mask関数とcall関数の引数にmask=Noneを指定すれば、maskは正しく動作するはず…
    #https://www.tensorflow.org/guide/keras/masking_and_padding　の　”Supporting masking in your custom layers”　にcompute_maskの解説あり
    def compute_mask(self, inputs, mask=None):
        return mask
    
    #https://www.tensorflow.org/guide/keras/masking_and_padding　の　”Writing layers that need mask information”　に引数にmask=Noneを指定する解説
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
        self.A , self.I , self.D, self.D_self = construct_graph.AandE_Matrix() 
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
  
  
