
#gerekli modulleri ve kütüphaneleri ekleyelim
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
#kullancağımız verileri (çiçek resimlerini) websitesinden tensorflow_datasets ile çekelim ve bir dosyaya kayıt edelim
#Not: Tüm resimler CC-BY lisanslıdır, yaratıcılar LICENSE.txt dosyasında listelenmiştir.
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
fname='flower_photos',
untar=True)
data_dir = pathlib.Path(data_dir)
#İndirdikten sonra (218MB), şimdi mevcut çiçek fotoğraflarının bir kopyasına sahip olmalısınız. 
# Toplam 3.670 resim var:
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
#Her dizin, o çiçek türünün resimlerini içerir. İşte bazı güller:


roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))
#Keras yardımcı programını kullanarak verileri yükleyin 
# ,Yükleyici için bazı parametreleri tanımlayın:
batch_size = 32
img_height = 180
img_width = 180

#Modelinizi geliştirirken bir doğrulama bölmesi kullanmak iyi bir uygulamadır. Görsellerin 
#%80'ini eğitim için ve %20'sini doğrulama için kullanacaksınız.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#Sınıf adlarını bu veri kümelerinde 
# class_names özniteliğinde bulabilirsiniz.
class_names = train_ds.class_names
print(class_names)


#Verileri görselleştirin
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
#Bu veri kümelerini model.fit geçirerek (bu öğreticide daha sonra gösterilmektedir) 
# kullanarak bir modeli eğitebilirsiniz. İsterseniz
# , veri kümesini manuel olarak yineleyebilir
# ve toplu görüntü alabilirsiniz:
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
#Verileri standartlaştırın
normalization_layer = tf.keras.layers.Rescaling(1./255)
#Bu katmanı kullanmanın iki yolu vardır. Dataset.map 
# arayarak veri kümesine uygulayabilirsiniz:
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
#Not: Piksel değerlerini [-1,1] olarak ölçeklemek istiyorsanız bunun yerine tf.keras.layers.
# Rescaling(1./127.5, offset=-1) yazabilirsiniz.
#Not: Görüntüleri daha önce tf.keras.utils.image_dataset_from_directory image_size bağımsız 
# değişkenini kullanarak yeniden boyutlandırmıştınız. Yeniden
# boyutlandırma mantığını modelinize de dahil etmek istiyorsanız,
# tf.keras.layers.Resizing katmanını kullanabilirsiniz.



AUTOTUNE = tf.data.AUTOTUNE
#Performans için veri kümesini yapılandırın
#Dataset.cache , görüntüleri ilk dönem boyunca diskten yüklendikten sonra bellekte tutar
#Dataset.prefetch , eğitim sırasında veri ön işleme ve model yürütme ile çakışır.

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#buraya kadar olan kısım veri çekme ve verileri kümelendirme işlemleriydi burdan sonraki kısımda 
#yapay zeka modelini eğiticez

num_classes = 5
#-------------------------------------------------------------------------------------------------#
#                                  #Modeli eğit                                   #
#Tamlık için, az önce hazırladığınız veri kümelerini 
# kullanarak basit bir modelin nasıl eğitileceğini göstereceksiniz.
#Sıralı model, her birinde 
#bir maksimum havuzlama katmanına 
#( tf.keras.layers.MaxPooling2D ) 
#sahip üç evrişim bloğundan ( tf.keras.layers.Conv2D ) oluşur. Üstünde 
#bir ReLU etkinleştirme işlevi ( 'relu' ) tarafından etkinleştirilen 
#128 birimlik tam bağlı bir katman ( tf.keras.layers.Dense ) vardır.
# Bu model hiçbir şekilde ayarlanmamıştır; amaç, az önce oluşturduğunuz
# veri kümelerini kullanarak size mekaniği göstermektir.
#-----------------------------------------------------------------------------------------------------------#
#keras katmanları oluşturulması işlemi
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
#oluşturduğumuz modeli compile ediyoruz
#tf.keras.optimizers.Adam optimizer ve tf.keras.losses.SparseCategoricalCrossentropy kaybı işlevini seçin. 
# Her eğitim dönemi için eğitim ve doğrulama doğruluğunu görüntülemek 
# için, metrics bağımsız değişkenini Model.compile .
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#Bu öğreticinin hızlı bir şekilde çalışması için yalnızca birkaç dönem için antrenman yapacaksınız.
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
#Yukarıdaki Keras ön işleme yardımcı programı — tf.keras.utils.image_dataset_from_directory — bir görüntü dizininden tf.data.Dataset oluşturmanın uygun bir yoludur.
#Daha hassas tahıl kontrolü için tf.data kullanarak kendi girdi işlem hattınızı yazabilirsiniz. 
#Bu bölüm, daha önce indirdiğiniz TGZ dosyasındaki dosya yollarından başlayarak tam da bunun nasıl yapılacağını gösterir.
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
#indirdiğimiz verileri bilgisayarımıza kaydediyoruz dosya olarak
for f in list_ds.take(5):
  print(f.numpy())
#Dosyaların ağaç yapısı, bir class_names listesi derlemek için kullanılabilir.  
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)
#Veri kümesini eğitim ve doğrulama kümelerine ayırıyoruz:
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

#Her veri kümesinin uzunluğunu aşağıdaki gibi yazdırabilirsiniz:
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())
#Bir dosya yolunu bir (img, label) çiftine dönüştüren kısa bir fonksiyon yazıyoruz:
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

#image, label çiftlerinden oluşan bir veri kümesi oluşturmak için Dataset.map kullanın:
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1): 
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
#performans için veri kümesini yapılandırın 
# Bu veri kümesiyle bir modeli eğitmek için aşağıdaki verileri isteyeceksiniz:
# #İyice karıştırılmak.
# #Toplu olarak Partiler mümkün olan en kısa sürede hazır olacak. Bu özellikler tf.data API kullanılarak eklenebilir.
def configure_for_performance(ds):
 ds = ds.cache()
 ds = ds.shuffle(buffer_size=1000)
 ds = ds.batch(batch_size)
 ds = ds.prefetch(buffer_size=AUTOTUNE)
 return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
#Bu veri kümesini daha önce oluşturduğunuza benzer şekilde görselleştirebilirsiniz
image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")
  
  #Modeli eğitmeye devam edelim:
  #simdi, yukarıdaki tf.keras.utils.image_dataset_from_directory 
  # tarafından oluşturulana benzer bir tf.data.Dataset
  # manuel olarak oluşturdunuz. 
  # Modeli onunla eğitmeye devam edebilirsiniz. 

  model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
  #TensorFlow Veri Kümelerini Kullanma:
  #Flowers veri kümesini daha önce diskten yüklediğiniz için, şimdi onu 
  # TensorFlow Veri Kümeleri ile içe aktaralım.
  (train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
  
  #çiçek kümesinin 5  sınıfı olması lazım
  num_classes = metadata.features['label'].num_classes
print(num_classes)

#veri kümesin bir görüntü alalım:
get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

#Daha önce olduğu gibi, performans için eğitim, doğrulama ve test setlerini
# gruplamayı, karıştırmayı ve yapılandırmayı unutmayın::
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)