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
        self.nodes = 49 #ノード数

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
