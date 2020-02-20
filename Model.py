#構築するモデル
class Model(tf.keras.Model):
    def __init__(self, opt_name, alpha, lambd, drop_rate):
        super(Model, self).__init__()
        self.lambd = lambd

        self.pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D([2,1], [2,1], "VALID"), input_shape=(201, 50, 3, 1))

        self.bn1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.gcn1  = tf.keras.layers.TimeDistributed(GcnLayer(3, 64, True, 25), input_shape=(201, 25, 3))
        self.cell1   =GcnLstmCell([25, 64],"4", True)#(self, units, name, Pooling=False,**kwargs):
        self.tgcn1 = tf.keras.layers.RNN(self.cell1, time_major=False, return_sequences=True)

        self.bn2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.gcn2  =  tf.keras.layers.TimeDistributed(GcnLayer(64, 128, True, 25))
        self.cell2   = GcnLstmCell([25, 128], "5", True)
        self.tgcn2 = tf.keras.layers.RNN(self.cell2, time_major=False, return_sequences=True)

        self.bn3 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.gcn3  =  tf.keras.layers.TimeDistributed(GcnLayer(128, 256, True, 25))    
        self.cell3  = GcnLstmCell([25, 256], "6", True)
        self.tgcn3 = tf.keras.layers.RNN(self.cell3, time_major=False, return_sequences=True)
    
        self.flatten1 = tf.keras.layers.Flatten()
        self.softmax  =  (DenseLayer(CLASS, True))#Trueならば、softmax層
        self.loss_object  = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        if opt_name=="Adam":
            self.opt             = tf.keras.optimizers.Adam(alpha)
        if opt_name=="Sgd":
            self.opt             = tf.keras.optimizers.SGD(alpha)

    def call(self, inputs, mask, Trainable):

        #勝手に定めたグラフプーリング(メモリに削減するために、書きました。)
        reshaped_inputs = tf.reshape(inputs, [-1, 201, 49, 3, 1])
        #ダミーノードを追加
        reshaped_inputs = tf.pad(reshaped_inputs, [[0,0], [0,0], [1,0], [0,0], [0,0]])
        reshaped_inputs = self.pool1(reshaped_inputs, mask= mask)
        reshaped_inputs = tf.reshape(reshaped_inputs, [-1, 201, 25, 3])

        bn1  = self.bn1(reshaped_inputs, training=Trainable, mask=mask)
        gcn1  = self.gcn1(bn1, mask=mask)
        tgcn1     = self.tgcn1(gcn1,mask=mask, training=Trainable)
        
        bn2  = self.bn2(tgcn1, training=Trainable, mask=mask)
        gcn2  = self.gcn2(bn2, mask=mask)
        tgcn2     = self.tgcn2(gcn2,mask=mask, training=Trainable)
    
        bn3  = self.bn3(tgcn2, training=Trainable, mask=mask)
        gcn3  = self.gcn3(bn3, Trainable, mask=mask)
        tgcn3     = self.tgcn3(gcn3,mask=mask, training=Trainable)
 
        flatten1 = self.flatten1(tgcn3)
        outputs  = self.softmax(flatten1)
        return outputs

    def train_step(self, images, labels, mask, Trainable):
        with tf.GradientTape() as tape:
            pred = self(images, mask, Trainable)
            loss  = self.loss_object(labels, pred)
            loss_l2 = 0.0
            for v in self.trainable_variables:
                 loss_l2 = loss_l2 + self.lambd*tf.reduce_sum(v**2)/2
            loss = loss + loss_l2
            loss  = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)
        grads   = tape.gradient(loss, self.trainable_variables)

        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        train_accuracy.update_state(labels, pred)

        return loss
    
    def test_step(self, images, labels, mask, Trainable):
        pred = self(images, mask, Trainable)
        loss =  self.loss_object(labels, pred)
        loss =  tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)
        test_accuracy.update_state(labels, pred)


        return loss

    #仕様書によると、experimental_run_v2は@tf.function内に書かないといけないみたい。
    @tf.function
    def distributed_train_step(self, images, labels, mask, Trainable):
            return  mirrored_strategy.experimental_run_v2(self.train_step, args=(images, labels, mask, Trainable))

    @tf.function
    def distributed_test_step(self, images, labels, mask, Trainable):
            return  mirrored_strategy.experimental_run_v2(self.test_step,  args=(images, labels, mask, Trainable))
    
    def reset(self):
        train_accuracy.reset_states()
        test_accuracy.reset_states()
  
