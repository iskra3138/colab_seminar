# Colab 사용법
### Colab 소개 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/1_Colab_%EC%86%8C%EA%B0%9C.ipynb)
- Colab FAQ를 통해 Colab이 뭔 지 간단히 살펴보고, VM 확인하고 변경하는 법에 대한 소개 

### Colab 기초 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/2_Colab_%EA%B8%B0%EC%B4%88.ipynb)
- 공식 소개 노트 중 'Working with Notebooks in Colaboratory'에 있는 내용 중 작업에 도움이 될만한 내용들 소개

### Colab 응용 1 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/3_colab_%EC%9D%91%EC%9A%A91.ipynb)
- 구글 드라이브와의 연동, 구글 클라우드 연동, 파일 업로드/다운로드 등 Storage 관련 내용에 대한 소개

### Colab 응용2 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/4_colab_%EC%9D%91%EC%9A%A92_tensorboard_in_notebooks.ipynb)
- Tensorboard 활용법 소개

### TPU 소개 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/5_tpu_%EC%86%8C%EA%B0%9C.ipynb)
- TPU에 대한 간단한 소개


# Colab 예제  
- 구글 BERT로 KorQuad 학습하는 예제
- git_source를 내려받아 python 파일로 실험하는 코드
- 출처 내용을 토대로 몇 가지 내용 수정 및 추가
- [출처](https://blog.nerdfactory.ai/2019/04/25/learn-bert-with-colab.html)
### 구글드라이브와 GPU 활용 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/colab%EC%97%90%EC%84%9C_KorQuad_%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B01_GPU.ipynb)

### 구글클라우드와 TPU 활용 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/colab%EC%97%90%EC%84%9C_KorQuad_%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B02_TPU.ipynb)
- GCP계정 필요  

# Colab TPU 예제 
- Cloud TPU 소개 자료 내 Colab 예제들 오류 수정, util 코드/주석 등 추가
- 코드 라인 단위 실험 예제들
- [출처](https://cloud.google.com/tpu/docs/colabs?hl=ko )

### TPU를 사용한 커스텀 학습: [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/cloud_tpu_custom_training.ipynb)

### Keras 및 TPU를 사용한 MNIST [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Keras_MNIST_TPU_Serving_GCP.ipynb)
- Serving시 GCP 계정 필요
- Keras API로 모델을 학습하고 GCP에 모델을 serving하는 예제

### Keras 및 TPU를 사용한 Fashion MNIST: 
- TPU 사용한 original [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Keras_Fashion_MNIST_TPU.ipynb)
- GPU 학습으로 변경한 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Keras_Fashion_MNIST_GPU.ipynb)
- CPU 학습으로 변경한 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Keras_Fashion_MNIST_CPU.ipynb)

### TPUEstimator로 MNIST 에스티메이터 포팅 by TPU [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/MNIST_Estimator_to_tpuestimator.ipynb)
- GCP계정필요
- TPUEstimator로 Google Cloud에서 MNIST학습하고 모델 Deploy하는 예제

### Estimator를 사용한 MNIST by GPU  [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/MNIST_Estimator_GPU.ipynb)
- Serving시 GCP계정 필요
- 바로 위 예제의 GPU 버전

### Autoencoder를 사용하여 TPU에서 임베딩 학습 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Train_embeddings_on_TPU_using_Autoencoder.ipynb)

### Keras를 사용한 꽃 분류 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Simple_Classification_Model_using_Keras_on_Colab_TPU.ipynb)
- Keras API를 사용하여 TPU로 학습하는 코드

### TPUEstimatior를 사용한 꽃 분류 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Simple_Classification_Model_using_TPUEstimator_on_Colab_TPU.ipynb)
- GCP 계정 필요
- TPUEstimator API를 사용하여 TPU로 학습하는 코드 

### Keras 및 TPU를 사용한 Shakespeare [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Predict_Shakespeare_with_Cloud_TPUs_and_Keras.ipynb)
- 다음 문자를 예측하는 Language Model

### TPUEstimator를 사용한 Shakespeare [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Predict_Shakespeare_with_Cloud_TPUs_and_TPUEstimator.ipynb)
- GCP 계정 필요
- 다음 문자를 예측하는 Language Model

### Kears를 사용한 사인 회귀 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/Simple_Regression_Model_Using_Keras_on_Colab_TPU_%5BFull_Sine_as_input%5D.ipynb)
- y=sin(x) 모델을 예측하는 예제

### Bert 및 Cloud TPU로 5분 내에 작업 미세 조정 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/BERT_End_to_End_(Fine_tuning_%2B_Predicting)_with_Cloud_TPU_Sentence_and_Sentence_Pair_Classification_Tasks.ipynb)
- GCP 계정 필요
- 사전 학습된 BERT 모델 위에 문장 및 문장 쌍 분류작업을 파인튜닝

### TF-GAN을 이용하여 TPU에서 생성적 적대 신경망(GAN)학습 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/TF_GAN_on_TPUs.ipynb)
- GCP 계정 필요
- TPU를 사용하여 CIFAR10데이터세트에서 GAN을 학습

# Pytorch Colab 메모장
### PyTorch/TPU MNIST 데모 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/PyTorch_TPU_MNIST_Training.ipynb)

### PyTorch/TPU ResNet18/CIFAR10 데모 [colab note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/PyTorch_TPU_ResNet18_CIFAR10_Training.ipynb)

### PyTorch/TPU ResNet50 추론 데모 [colab  note](https://colab.research.google.com/github/iskra3138/colab_seminar/blob/master/%5BPT_Devcon_2019%5D_PyTorch_TPU_ResNet50_Inference.ipynb)
