## introduction
`NNstreamer`는 오래된 CPI와 적은 메모리 용량을 가지고 있는 컴퓨터 기반의 서버를 지원하기 위해 만들어진 오픈소스 라이브러리이다. Gstreamer의 개발자들에게는 nn모델을 쉽고 효율적으로 적용할 수 있도록 도와준다. neuarl network 개발자들에게는 stream pipeline과 그들의 filter를 효율적으로 관리할 수 있도록 도와준다. 

    - 이 라이브러리는 gstreamer streams에게 nn framework connectivities(ex. tensorflow, caffe ,..)를 제공한다.
    - nn 모델을 효율적이로 유연한 streaming management를 원하는데 nnstreamer가 이를 지원해 줄 것이다. 
    - neuarl network model을 media filter 또는 converter로 사용할 수 있다. 
    - single stream instance에서 여러개의 neural network 모델을 사용할 수 있도록 해준다. 
    - neural network 모델에게 여러개의 source들을 사용할 수 있도록 해준다. 

## how to build
라이브러리 설치 방법과 example 빌드 방법은 issue로 등록해두었습니다. 