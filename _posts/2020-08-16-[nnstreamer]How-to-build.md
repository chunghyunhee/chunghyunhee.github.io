# basic terminology 

## build? compile? link?
가장 기본적인 의미로 프로그래밍 언어를 기계어로 바꿔주는 과정이다. native code에서 *.o파일을 만드는 것을 `컴파일`이라고 한다. 기능별로 분할된 여러개의 native code들이 있을 때 각각을 *.o파일로 컴파일 하고 다수의 *.o파일을 하나의 *.exe파일을 만드는 것을 `링크`라고 한다. 컴파일과 링크를 합쳐 실행가능한 하나의 파일을 만드는 과정이 `빌드`라고 한다. 

## make?
유닉스 계열의 운영체제에서 주로 사용되는 build도구이다. 여러 파일들끼리의 의존성과 각 파일에 필요한 명령을 정의함으로써 프로그램을 compile할 수 있으며 프로그램을 만들 수 있는 과정을 서술 할 수 있는 표준적인 문법을 가진다. 만약 build도구 없이 IDE를 통해 build하고 deploy할 때의 가장 큰 문제는 해당 프로그램의 운영환경이 배포자의 pc에 의해 결정된다는 점이다. 

## makefile?
make를 실행하기 전에 프로젝트의 목록 및 compile 및 link 규칙을 만들어야 한다. 이런 규칙을 명시하는 파일을 만들고 이를 Makefile이라고 한다. 

```bash
helloworld:
	gcc main.c -o helloworld

install: helloworld
	install -m 0755 helloworld /usr/local/bin
```
## configure?
configure script는 개발 중인 프로그램을 각기 다른 수많은 컴퓨터들에서 실행할 수 있도록 도와주도록 설계된 실행스크립트이다. 소스코드부터 컴파일하기 직전에 사용자 컴퓨터의 라이브러리의 존재 여부를 확인하고 연결시켜준다. 

## cmake?
cmake는 빌드 도구중 하나이다. 정확히는 build system에서 필요로 하는 파일을 생성하는 것이 그 목적이다. 프로그램을 빌드하는데 있는 것은 아니다. cmake를 활용하여 프로젝트를 관리하고자 한다면 필요/의도에 맞게 CmakeLists.txt파일을 배치해야 한다. 

```bash
$ sudo apt install tree
```

원하는 폴더를 확인한다면 그래프를 그려줍니다. 

```bash
$ tree ./gst-docs
```

결과는 아래와 같은 형태이다. 

```bash
├── scripts
│   ├── RELEASE_README.md
│   ├── generate_sitemap.py
│   └── release.py
├── sitemap.txt
└── theme
    └── extra
        ├── css
        │   └── extra_frontend.css
        ├── images
        │   ├── api-reference.svg
```

재귀적으로 CmakeLists.txt를 위치시키는 모습을 아래와 같습니다. 
```bash
$ tree ./small-project/
./small-project         # Project root folder
├── CMakeLists.txt      # <---  Root CMake
├── include             # header files
│   └── ... 
├── module1             # sub-project
│   ├── CMakeLists.txt
│   └── ... 
├── module2             # sub-project
│   ├── CMakeLists.txt
│   └── ... 
└── test                # sub-project
    ├── CMakeLists.txt
    └── ... 
```



