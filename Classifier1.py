{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Classifier1.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyNKiWij4GN0gAxFciv8qFxu"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "j94lTx_86qEL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.image as im\n",
        "import boto3\n",
        "import cv2 as cv2\n",
        "import tempfile\n",
        "import PIL.Image"
      ],
      "execution_count": 13,
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
        "ACCESS_KEY = 'AKIAIN3MGH3LKJXWCRDA'\n",
        "SECRET_KEY = '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'"
      ],
      "execution_count": 14,
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
      "execution_count": 15,
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
      "execution_count": 16,
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
        "for i in bucket.objects.filter(Prefix='Denomination_Original/'):\n",
        "  if i.key !=ignore:\n",
        "   l.append(i.key)\n",
        "  else:\n",
        "    continue\n",
        "print('Total Images in S3 bucket: '+str(len(l)))\n",
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
        "print('Total Images is : '+str(total))\n",
        "print('Total Images Shape : '+str(final_inp.shape))"
      ],
      "execution_count": null,
      "outputs": []
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
        "print(X.shape)\n",
        "print(X)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Syyj1GKBbzfS",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#model=tf.keras.applications.InceptionResNetV2(include_top=False,input_shape=(img_width,img_height,3),classes=2,classifier_activation='softmax',)\n",
        "#model.summary()"
      ],
      "execution_count": 19,
      "outputs": []
    }
  ]
}