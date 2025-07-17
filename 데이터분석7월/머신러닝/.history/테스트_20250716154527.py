from konlpy.tag import Okt
okt = Okt()
print(okt.morphs("자연어 처리는 재밌습니다!"))

import tensorflow as tf
print("✅ TensorFlow version:", tf.__version__)
print("✅ GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))

