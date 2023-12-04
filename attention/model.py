from keras.layers import Input, Dense, Layer, Dropout
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.losses import BinaryFocalCrossentropy

class AttentionPooling(Layer):

    """Implementation of the attention-based Deep MIL layer"""

    def __init__(
            self,
            weights_dim=512,
            V_init='uniform',
            w_init='uniform',
            kernel_regularizer=None,
            use_gated_attention=False,
            **kwargs,
    ):
        
        super(AttentionPooling, self).__init__(**kwargs)

        self.weights_dim = weights_dim
        self.V_init = V_init
        self.w_init = w_init
        self.kernel_regularizer = kernel_regularizer
        self.use_gated_attention = use_gated_attention

    def build(self, input_shape):

        """Creates the layer weights
        input_shape: (batch_size, time_steps, input_dim)"""

        time_steps = input_shape[1]
        input_dim = input_shape[2]

        self.V_weights = self.add_weight(
            name='V_weights',
            shape=(input_dim, self.weights_dim),
            initializer=self.V_init,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.w_weights = self.add_weight(
            name='w_weights',
            shape=(self.weights_dim, 1),
            initializer=self.w_init,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        if self.use_gated_attention:

            self.U_weights = self.add_weight(
                name='U_weights',
                shape=(input_dim, self.weights_dim),
                initializer=self.V_init,
                regularizer=self.kernel_regularizer,
                trainable=True
            )

        super(AttentionPooling, self).build(input_shape)

    def call(self,inputs):

        """Computes the attention weights
        inputs: (batch_size, time_steps, input_dim)"""

        if self.use_gated_attention:

            V = K.tanh(K.dot(inputs, self.V_weights))
            U = K.sigmoid(K.dot(inputs, self.U_weights))
            att = K.dot(V * U, self.w_weights)

        else:

            att = K.dot(K.tanh(K.dot(inputs, self.V_weights)), self.w_weights)

        att = K.squeeze(att, axis=-1)
        att = K.softmax(att)
        att = K.batch_dot(att, inputs, axes=1)

        return att
    
    def get_config(self):

        config = super().get_config()

        config.update({
            'weights_dim': self.weights_dim,
            'V_init': self.V_init,
            'w_init': self.w_init,
            'kernel_regularizer': self.kernel_regularizer,
            'use_gated_attention': self.use_gated_attention,
        })    

        return config
    
class FeatureSelectionModule(Layer):

    '''Module to perform feature selection with following steps:
    Non-linear transformation
    Batch-wise attenuation
    Feature Mask Normalization'''

    def __init__(
        self,
        nb_units,
        kernel_regularizer=None,
        **kwargs,
    ):
        self.nb_units = nb_units
        self.kernel_regularizer = kernel_regularizer
        super(FeatureSelectionModule, self).__init__(**kwargs)

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.W_1 = self.add_weight(
            name='W_1',
            shape=(input_dim, self.nb_units),
            initializer='uniform',
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.W_2 = self.add_weight(
            name='W_2',
            shape=(self.nb_units, input_dim),
            initializer='uniform',
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.b_1 = self.add_weight(
            name='b_1',
            shape=(self.nb_units,),
            initializer='uniform',
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.b_2 = self.add_weight(
            name='b_2',
            shape=(input_dim,),
            initializer='uniform',
            regularizer=self.kernel_regularizer,
            trainable=True
        )

    def call(self, inputs):

        # Non-linear transformation
        x = K.dot(inputs, self.W_1) + self.b_1
        x = K.tanh(x)
        x = K.dot(x, self.W_2) + self.b_2

        # Batch-wise attenuation
        x = K.mean(x, axis=0)

        # Feature Mask Normalization
        x = K.softmax(x, axis=0)

        return x
    
    def get_config(self):

        config = super().get_config()

        config.update({
            'nb_units': self.nb_units,
        })    

        return config
    
class LearningModule(Layer):

    '''Module to perform learning after pooling and feature selection'''

    def __init__(
        self,
        output_dim=1,
        **kwargs,
    ):
        self.output_dim = output_dim
        super(LearningModule, self).__init__(**kwargs)

    def build(self, input_shape):

        # self.dense1 = Dense(64, activation='leaky_relu')
        self.dense2 = Dense(32, activation='leaky_relu')
        self.dropout = Dropout(0.3)
        self.dense3 = Dense(self.output_dim, activation='sigmoid')

    def call(self, inputs):

        # x = self.dense1(inputs)
        x = self.dense2(inputs)
        x = self.dropout(x)
        x = self.dense3(x)

        return x
    
    def get_config(self):

        config = super().get_config()

        config.update({
            'output_dim': self.output_dim,
        })    

        return config

def create_model(
        input_dim,
        output_dim=1,
        attention_units=128,
        feature_selection_units=512,
        V_init='uniform',
        w_init='uniform',
        learning_rate=1e-3,
        attention_kernel_regularizer=None,
        feature_selection_kernel_regularizer=None,
        use_gated_attention=False,
):

    """Creates the Deep MIL model
    input_dim: dimension of the input data
    output_dim: dimension of the output data
    weights_dim: dimension of the attention weights
    V_init: initialization of the V weights
    w_init: initialization of the w weights"""

    inputs = Input(shape=input_dim)

    x_pooled = AttentionPooling(
        attention_units,
        V_init,
        w_init,
        kernel_regularizer=attention_kernel_regularizer,
        use_gated_attention=use_gated_attention,
    )(inputs)

    feature_mask = FeatureSelectionModule(
        feature_selection_units,
        kernel_regularizer=feature_selection_kernel_regularizer,
    )(x_pooled)

    x_pooled_fs = x_pooled * feature_mask

    output = LearningModule(output_dim)(x_pooled_fs)
    
    model = Model(inputs=inputs, outputs=output)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[AUC()],
    )
    model.build(input_shape=input_dim)

    return model