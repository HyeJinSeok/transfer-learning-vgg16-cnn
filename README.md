# VGG 전이학습을 통한 고양이·개 이미지 분류

<br>

이미지 분류(Image Classification)는 컴퓨터비전 분야에서 가장 기본적이면서도 중요한 문제다. <br>

특히 '개 vs 고양이 분류'는 대표적인 **이진 분류(binary classification)** 문제로, 많은 연구에서 벤치마크로 활용된다. <br>

하지만 딥러닝 모델을 처음부터 학습(Training from Scratch)하려면, 수십만 장 이상의 대규모 데이터와 긴 학습시간이 필요하다. <br>

또한 데이터가 충분하지 않은 경우에는 <strong>과적합(overfitting)</strong>이 쉽게 발생하고, 학습된 모델의 일반화 성능도 떨어질 수 있다. <br>

이러한 한계를 해결하기 위한 방법으로 **전이학습(Transfer Learning)** 기법이 있다. <br>

전이학습은 대규모 데이터셋에서 미리 학습된 **신경망의 가중치**를 가져와, 새로운 문제에 맞게 <strong>일부 레이어만 재학습(fine-tuning)</strong>한다. <br>

이를 통해 학습 시간을 크게 단축하고, 작은 데이터셋에서도 안정적인 성능을 낼 수 있다. <br>

이번 프로젝트에서는 VGG16 모델을 기반으로 전이학습을 수행한다.

<br>

## VGG16 (Visual Geometry Group 16-layer CNN)
