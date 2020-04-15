# Unit Block of CovXNet:

def unit_block(xin, ch, dil_range, n):
  for i in range(n):
    x1 = Conv2D(ch*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(xin)
    x1 = BatchNormalization()(x1)
  
    a = []

    for i in range(1, dil_range+1):
      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
      temp = BatchNormalization()(temp)
      a.append(temp)

    x = Concatenate(axis= -1)(a)
    x = Conv2D(ch, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([x, xin])

    xin = x
  
  return x


"""
           ch           = Number of Channels
           dil_range    = Dialation Range
           n            = Depth of the Network
           xin          = Input
           
"""


# Shifter Block of CovXNet:

def shifter_block(xin, ch):
    x1 = Conv2D(ch*4, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(xin)
    x1 = BatchNormalization()(x1)

    x = DepthwiseConv2D( kernel_size=(3,3), strides = (2,2), dilation_rate = (1,1), padding = 'same', activation= 'relu')(x1)
    x = BatchNormalization()(x)

    x = Conv2D(ch*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    return x
  
  
"""
           xin          = Input
           ch           = Number of Channel
           
"""
  
# Network of CovXNet: 

def Network(input_shape, nb_class, depth):
  xin = Input(shape= input_shape)

  x = Conv2D(16, kernel_size = (5,5), strides= (1,1), padding = 'same', activation='relu')(xin)
  x = BatchNormalization()(x)

  x = Conv2D(32, kernel_size = (3,3), strides= (2,2), padding = 'same', activation='relu')(xin)
  x = BatchNormalization()(x)
  
  x = unit_block( x, 32, 5, depth)
  x = shifter_block(x, 32)

  x = unit_block( x, 64, 4, depth)
  x = shifter_block(x, 64)

  x = unit_block( x, 128, 3, depth)
  x = shifter_block(x, 128)

  x = unit_block( x, 256, 2, depth)

  x = GlobalAveragePooling2D()(x)

  x = Dense(64, activation='relu')(x)
  x = Dense(nb_class, activation= 'sigmoid')(x)

  model = Model(xin, x)

  model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  return model

"""
           xin          = Input
           ch           = Number of Channel
           nb_class     = Number of Class
           depth        = Depth of the Network
           
"""
