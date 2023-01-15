from keras_preprocessing.image import load_img , img_to_array
from keras.models import load_model
import numpy as np
mymodel=load_model('model.h5')


test_image= load_img(r"D:\new\base\test_set\with_mask\maksssksksss1.png", target_size=(150,150,3))
test_image= img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
pred = mymodel.predict(test_image)[0][0]
pred = list(i for i in str(pred) if i!='0' or i!='.')
pred = int(pred[0])
print(pred)
