from keras.layers import Input, Dense, Layer, Dropout
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.metrics import AUC

class Attention(Layer):

    """Implementation of the attention-based Deep MIL layer"""

    def __init__(
            self,
            weights_dim=512,
            V_init='uniform',
            w_init='uniform',
            kernel_regularizer=None,
            use_gated_attention=False,
            **kwargs
    ):
        
        super(Attention, self).__init__(**kwargs)

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

        super(Attention, self).build(input_shape)

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

def create_model(
        input_dim,
        output_dim=1,
        weights_dim=512,
        V_init='uniform',
        w_init='uniform',
        learning_rate=1e-3,
        kernel_regularizer=None,
        use_gated_attention=False,
):

    """Creates the Deep MIL model
    input_dim: dimension of the input data
    output_dim: dimension of the output data
    weights_dim: dimension of the attention weights
    V_init: initialization of the V weights
    w_init: initialization of the w weights"""

    inputs = Input(shape=input_dim)
    x = Attention(
        weights_dim,
        V_init,
        w_init,
        kernel_regularizer=kernel_regularizer,
        use_gated_attention=use_gated_attention,
    )(inputs)
    x = Dropout(0.4)(x)
    x = K.batch_dot(x, inputs, axes=1)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='sigmoid')(x)
    x = Dropout(0.4)(x)
    x = Dense(output_dim, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[AUC()],
    )

    return model