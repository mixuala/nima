# TF Slim Learning Pipeline

## Data Pipeline

Dataset

DatasetDataProvider


## flowers data pipeline
```
# create TFRecords
dataset_utils.download_and_uncompress_tarball()
  datasets.download_and_convert_flowers()
    dataset_utils.download_and_uncompress_tarball()
    _get_filenames_and_classes()
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # e.g.  { daisy:0, ...}
    ### Divide into train and test:
    _convert_dataset('train',...)
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    _clean_up_temporary_files()
### Output: labels_to_class_names, TFRecord datasets in dataset_dir

```

Output:
    ```
    labels.txt
    flowers_{[train,validation]}_{i}-of-{m}.tfrecord
    ```

labels.txt:
    0:daisy
    1:dandelion
    2:roses
    3:sunflowers
    4:tulips


## vgg preprocessing
```

image = _aspect_preserving_resize(image, resize_side)
image = _random_crop([image], output_height, output_width)[0]

### restore RGB channels
image.set_shape([output_height, output_width, 3])

image = tf.to_float(image)

### randomly flip image
image = tf.image.random_flip_left_right(image)

_mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

```

## inception preprocessing
```
# load_batch
data_provider = slim.dataset_data_provider.DatasetDataProvider()
image_raw, label = data_provider.get(['image', 'label'])

# preprocess image, inceptions_preprocessing
image = inception_preprocessing.preprocess_image()
  preprocess_for_train()
  
  ### distort 1 image for training
  distorted_image.distorted_bounding_box_crop()
  
  ### restore RGB channels
  distorted_image.set_shape([None, None, 3])
  
  ### resize distorted image (with randomly sampled method)
  tf.image.resize_images()
  
  ### randomly flip image
  tf.image.random_flip_left_right()

```



# tf common patterns
```
if not tf.gfile.Exists(flowers_data_dir):
    tf.gfile.MakeDirs(flowers_data_dir)
```

```
labels_to_class_names = dict(zip(range(len(class_names)), class_names))
# outputs: {0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips}
```