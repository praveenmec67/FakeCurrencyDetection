{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Classifier1.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOYhNz37w8hWKpkLduvsily",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/praveenmec67/FakeCurrencyDetection/blob/master/Classifier1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "j94lTx_86qEL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.image as im\n",
        "import boto3\n",
        "import os\n",
        "import cv2 as cv2\n",
        "import tempfile\n",
        "from sklearn.model_selection import train_test_split"
      ],
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U1gG4IKP63fr",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "img_width=224\n",
        "img_height=224\n",
        "channels=3\n",
        "l=[]\n",
        "empty=np.empty([1,224,224,3])\n",
        "\n",
        "ACCESS_KEY = 'AKIAJAGTRUKOQSQK3WSA'\n",
        "SECRET_KEY = '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'\n"
      ],
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PcwNoynyTFw5",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def visualize(img,blur,medianfil,rotate,brightness,contrast):\n",
        "\n",
        "   fig = plt.figure(figsize=(20,20))\n",
        "   fig.subplots_adjust(wspace=0.5)\n",
        "\n",
        "  \n",
        "   plt.subplot(1,7,1)\n",
        "   plt.title('Original')\n",
        "   plt.imshow(img)\n",
        "\n",
        "   plt.subplot(1,7,2)\n",
        "   plt.title('Blur')\n",
        "   plt.imshow(blur)\n",
        "\n",
        "\n",
        "   plt.subplot(1,7,3)\n",
        "   plt.title('MedianFilter')\n",
        "   plt.imshow(medianfil)\n",
        "\n",
        "   plt.subplot(1,7,4)\n",
        "   plt.title('Rotate')\n",
        "   plt.imshow(rotate)\n",
        "\n",
        " #  plt.subplot(1,7,5)\n",
        " #  plt.title('GrayScale')\n",
        " #  plt.imshow(gray)\n",
        "\n",
        "   plt.subplot(1,7,5)\n",
        "   plt.title('Brightness')\n",
        "   plt.imshow(brightness)\n",
        "\n",
        "   plt.subplot(1,7,6)\n",
        "   plt.title('Contrast')\n",
        "   plt.imshow(contrast)"
      ],
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-yHQDs9nTI4N",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def image_aug(img):\n",
        "  \n",
        "  \n",
        "  rows,cols = img.shape[0],img.shape[1]\n",
        "  b,c = np.random.uniform(0.2,0.6),np.random.uniform(0.2,0.6)\n",
        "\n",
        "  img_org=img\n",
        "  img_org=img.reshape(1,img_width,img_height,channels)\n",
        "  img_org=tf.convert_to_tensor(img_org,dtype=tf.uint8)\n",
        "\n",
        "\n",
        "\n",
        "  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)\n",
        "  #gray=gray.reshape(1,img_width,img_height,channels)\n",
        "  #gray=tf.convert_to_tensor(gray,dtype=tf.uint8)\n",
        "  \n",
        "  \n",
        "  blur=cv2.blur(img,(5,5))\n",
        "  blur=blur.reshape(1,img_width,img_height,channels)\n",
        "  blur=tf.convert_to_tensor(blur,dtype=tf.uint8)\n",
        "\n",
        "\n",
        "  medianfil=cv2.medianBlur(img,5)\n",
        "  medianfil=medianfil.reshape(1,img_width,img_height,channels)\n",
        "  medianfil=tf.convert_to_tensor(medianfil,dtype=tf.uint8)\n",
        "\n",
        "\n",
        "  brightness=tf.image.adjust_brightness(img,b)\n",
        "  brightness=tf.reshape(img,[1,img_width,img_height,channels])\n",
        "\n",
        "\n",
        "  contrast=tf.image.adjust_contrast(img,c)\n",
        "  contrast=tf.reshape(img,[1,img_width,img_height,channels])\n",
        "\n",
        "\n",
        "\n",
        "  for i in np.random.randint(0,360,1):\n",
        "    M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)\n",
        "    rotate = cv2.warpAffine(img,M,(cols,rows))\n",
        "    rotate=rotate.reshape(1,rotate.shape[0],rotate.shape[1],rotate.shape[2])\n",
        "    rotate=tf.convert_to_tensor(rotate,dtype=tf.uint8)\n",
        "\n",
        "\n",
        "    img_aug=np.concatenate([img_org,blur,medianfil,brightness,contrast,rotate],axis=0)\n",
        "    return img_aug\n",
        " \n",
        "\n",
        "\n",
        "  \n",
        "\n",
        "  #visualize(img,blur,medianfil,rotate,brightness,contrast)"
      ],
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dFLqGjuSTMYR",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "\n",
        "s3=boto3.resource('s3',aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)\n",
        "bucket=s3.Bucket('fakedetection')\n",
        "ignore='Denomination_Original/'\n",
        "\n",
        "\n",
        "for i in bucket.objects.filter(Prefix='Denomination_Original/'):\n",
        "  if i.key !=ignore:\n",
        "   l.append(i.key)\n",
        "  else:\n",
        "    continue\n",
        "print('Total Images in S3 bucket: '+str(len(l)))\n",
        "\n",
        "\n",
        "total=0\n",
        "for j in l:\n",
        "  if total==len(l):\n",
        "    break\n",
        "  else:\n",
        "    name='img_'+str(total)\n",
        "    key=j\n",
        "    obj=bucket.Object(key)\n",
        "    tmp = tempfile.NamedTemporaryFile()\n",
        "\n",
        "    with open(tmp.name, 'wb') as f:\n",
        "      obj.download_fileobj(f)\n",
        "      img=im.imread(tmp.name)\n",
        "      img=cv2.resize(img,(img_width,img_height),interpolation=cv2.INTER_LINEAR)\n",
        "      \n",
        "      print(j +' : '+str(img.shape))\n",
        "      images=image_aug(img)\n",
        "      final_inp=np.append(empty,images,0)\n",
        "      empty=final_inp\n",
        "      total=total+1\n",
        "      \n",
        "\n",
        "final_inp=np.delete(final_inp,[0],0)\n",
        "\n",
        "\n",
        "print('Total Images is : '+str(total))\n",
        "print('Total Images Shape : '+str(final_inp.shape))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lvykpfHugXe-",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        },
        "outputId": "ebea0754-6e80-4d7d-cc1b-e537db90c3be"
      },
      "source": [
        "s3=boto3.resource('s3',aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)\n",
        "bucket=s3.Bucket('fakedetection')\n",
        "ignore='Labels/'\n",
        "list=[]\n",
        "for i in bucket.objects.filter(Prefix='Labels/'):\n",
        "  if i.key!=ignore:\n",
        "    list.append(i.key)\n",
        "    print(list)\n",
        "  else:\n",
        "    continue\n",
        "for j in list:\n",
        "    key=j\n",
        "    print(key)\n",
        "    obj=bucket.Object(key)\n",
        "    print(obj)\n",
        "    tmp = tempfile.NamedTemporaryFile()\n",
        "\n",
        "    with open(tmp.name, 'wb') as f:\n",
        "      obj.download_fileobj(f)\n",
        "      y=pd.read_csv(tmp.name).iloc[:,1].values\n",
        "      print(y)\n",
        "\n",
        "  "
      ],
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "['Labels/Train_labels.csv']\n",
            "Labels/Train_labels.csv\n",
            "s3.Object(bucket_name='fakedetection', key='Labels/Train_labels.csv')\n",
            "[1 1 1 ... 0 0 0]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vuopJ38_3clL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X=tf.reshape(final_inp,shape=[1002,(224*224*3)])\n",
        "X=X/255.\n",
        "train_images=X[:802,:]\n",
        "test_images=X[802:,:]\n",
        "train_labels=y[:802]\n",
        "test_labels=y[802:]\n",
        "\n"
      ],
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Syyj1GKBbzfS",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 786
        },
        "outputId": "2f315636-79f8-4f77-d0e6-69436642371c"
      },
      "source": [
        "model=tf.keras.applications.InceptionResNetV2(include_top=False,input_shape=(img_width,img_height,3),classes=2,classifier_activation='softmax',)\n",
        "#model.summary()\n",
        "model.compile(optimizer='Adam',loss='categorical_crossentropy')\n",
        "result=model.fit(train_images,train_labels,epochs=3)"
      ],
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/3\n",
            "WARNING:tensorflow:Model was constructed with shape (None, 224, 224, 3) for input Tensor(\"input_2:0\", shape=(None, 224, 224, 3), dtype=float32), but it was called on an input with incompatible shape (None, 150528).\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-60-9472dde60e12>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;31m#model.summary()\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcompile\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moptimizer\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'Adam'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'categorical_crossentropy'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mresult\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_images\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mtrain_labels\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mepochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/training.py\u001b[0m in \u001b[0;36m_method_wrapper\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m     64\u001b[0m   \u001b[0;32mdef\u001b[0m \u001b[0m_method_wrapper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     65\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_in_multi_worker_mode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m  \u001b[0;31m# pylint: disable=protected-access\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 66\u001b[0;31m       \u001b[0;32mreturn\u001b[0m \u001b[0mmethod\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     67\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     68\u001b[0m     \u001b[0;31m# Running inside `run_distribute_coordinator` already.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/training.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)\u001b[0m\n\u001b[1;32m    846\u001b[0m                 batch_size=batch_size):\n\u001b[1;32m    847\u001b[0m               \u001b[0mcallbacks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_train_batch_begin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 848\u001b[0;31m               \u001b[0mtmp_logs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    849\u001b[0m               \u001b[0;31m# Catch OutOfRangeError for Datasets of unknown size.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    850\u001b[0m               \u001b[0;31m# This blocks until the batch has finished executing.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    578\u001b[0m         \u001b[0mxla_context\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mExit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    579\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 580\u001b[0;31m       \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    581\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    582\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mtracing_count\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_tracing_count\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36m_call\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    625\u001b[0m       \u001b[0;31m# This is the first call of __call__, so we have to initialize.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    626\u001b[0m       \u001b[0minitializers\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 627\u001b[0;31m       \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_initialize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0madd_initializers_to\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0minitializers\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    628\u001b[0m     \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    629\u001b[0m       \u001b[0;31m# At this point we know that the initialization is complete (or less\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36m_initialize\u001b[0;34m(self, args, kwds, add_initializers_to)\u001b[0m\n\u001b[1;32m    504\u001b[0m     self._concrete_stateful_fn = (\n\u001b[1;32m    505\u001b[0m         self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access\n\u001b[0;32m--> 506\u001b[0;31m             *args, **kwds))\n\u001b[0m\u001b[1;32m    507\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    508\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0minvalid_creator_scope\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0munused_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0munused_kwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m_get_concrete_function_internal_garbage_collected\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   2444\u001b[0m       \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2445\u001b[0m     \u001b[0;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2446\u001b[0;31m       \u001b[0mgraph_function\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_maybe_define_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2447\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mgraph_function\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2448\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m_maybe_define_function\u001b[0;34m(self, args, kwargs)\u001b[0m\n\u001b[1;32m   2775\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2776\u001b[0m       \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_function_cache\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmissed\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcall_context_key\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2777\u001b[0;31m       \u001b[0mgraph_function\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_create_graph_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2778\u001b[0m       \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_function_cache\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprimary\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcache_key\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgraph_function\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2779\u001b[0m       \u001b[0;32mreturn\u001b[0m \u001b[0mgraph_function\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m_create_graph_function\u001b[0;34m(self, args, kwargs, override_flat_arg_shapes)\u001b[0m\n\u001b[1;32m   2665\u001b[0m             \u001b[0marg_names\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0marg_names\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2666\u001b[0m             \u001b[0moverride_flat_arg_shapes\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0moverride_flat_arg_shapes\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2667\u001b[0;31m             capture_by_value=self._capture_by_value),\n\u001b[0m\u001b[1;32m   2668\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_function_attributes\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2669\u001b[0m         \u001b[0;31m# Tell the ConcreteFunction to clean up its graph once it goes out of\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/func_graph.py\u001b[0m in \u001b[0;36mfunc_graph_from_py_func\u001b[0;34m(name, python_func, args, kwargs, signature, func_graph, autograph, autograph_options, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, override_flat_arg_shapes)\u001b[0m\n\u001b[1;32m    979\u001b[0m         \u001b[0m_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moriginal_func\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtf_decorator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munwrap\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpython_func\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    980\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 981\u001b[0;31m       \u001b[0mfunc_outputs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpython_func\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mfunc_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfunc_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    982\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    983\u001b[0m       \u001b[0;31m# invariant: `func_outputs` contains only Tensors, CompositeTensors,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36mwrapped_fn\u001b[0;34m(*args, **kwds)\u001b[0m\n\u001b[1;32m    439\u001b[0m         \u001b[0;31m# __wrapped__ allows AutoGraph to swap in a converted function. We give\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    440\u001b[0m         \u001b[0;31m# the function a weak reference to itself to avoid a reference cycle.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 441\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mweak_wrapped_fn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__wrapped__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    442\u001b[0m     \u001b[0mweak_wrapped_fn\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mweakref\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mref\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mwrapped_fn\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    443\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/func_graph.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    966\u001b[0m           \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m  \u001b[0;31m# pylint:disable=broad-except\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    967\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mhasattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"ag_error_metadata\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 968\u001b[0;31m               \u001b[0;32mraise\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mag_error_metadata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto_exception\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    969\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    970\u001b[0m               \u001b[0;32mraise\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: in user code:\n\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/training.py:571 train_function  *\n        outputs = self.distribute_strategy.run(\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/distribute/distribute_lib.py:951 run  **\n        return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/distribute/distribute_lib.py:2290 call_for_each_replica\n        return self._call_for_each_replica(fn, args, kwargs)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/distribute/distribute_lib.py:2649 _call_for_each_replica\n        return fn(*args, **kwargs)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/training.py:531 train_step  **\n        y_pred = self(x, training=True)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/base_layer.py:927 __call__\n        outputs = call_fn(cast_inputs, *args, **kwargs)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/network.py:719 call\n        convert_kwargs_to_constants=base_layer_utils.call_context().saving)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/network.py:888 _run_internal_graph\n        output_tensors = layer(computed_tensors, **kwargs)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/base_layer.py:886 __call__\n        self.name)\n    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/input_spec.py:180 assert_input_compatibility\n        str(x.shape.as_list()))\n\n    ValueError: Input 0 of layer conv2d_203 is incompatible with the layer: expected ndim=4, found ndim=2. Full shape received: [None, 150528]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y8XDOxT1lAph",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "plt.plot(result.result['accuracy'], label='accuracy')\n",
        "plt.plot(result.result['val_accuracy'], label = 'val_accuracy')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.ylim([0.5, 1])\n",
        "plt.legend(loc='lower right')\n",
        "\n",
        "test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}