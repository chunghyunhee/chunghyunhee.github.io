## intro
기본적으로 nnstreamer는 Gstreamer를 지원하는 plug-in이므로 gstreamer tutorial을 먼저 공부하고  nnstreamer contribution을 시작해야 한다. 

tutorial(https://gstreamer.freedesktop.org/documentation/tutorials/basic/hello-world.html?gi-language=c)을 참고하여 공부했다. 

## 목표
새로운 라이브러리를 접할 때는 hello world를 확인하면서 첫 시작을 한다. 하지만 Gstreamer는 multimedia framework이기 때문에 hello world대신에 비디오를 실행시니다. 

## 실행시켜보기 

## getting started tutorial's source code
1. clone repo
```bash
git clone https://gitlab.freedesktop.org/gstreamer/gst-docs
```
2. 실행할 폴더로 이동
```bash
$ cd gst-docs/examples/tutorials/
```

3. 튜토링러 파일을 실행하고 영상이 실행되어야 한다. 
```bash
$ gcc basic-tutorial-1.c -o basic-tutorial-1 `pkg-config --cflags --libs gstreamer-1.0`
```

### understand
basic-tuturials-1.c파일은 다음과 같습니다. 
```c
//basic-tuturials-1.c
#include <gst/gst.h>

int
main (int argc, char *argv[])
{
  GstElement *pipeline;
  GstBus *bus;
  GstMessage *msg;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  /* Build the pipeline */
  pipeline =
      gst_parse_launch
      ("playbin uri=https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm",
      NULL);

  /* Start playing */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait until error or EOS */
  bus = gst_element_get_bus (pipeline);
  msg =
      gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE,
      GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

  /* Free resources */
  if (msg != NULL)
    gst_message_unref (msg);
  gst_object_unref (bus);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  return 0;
}
```

다음의 code를 실행합니다. 
```bash
$ gcc basic-tutorial-1.c -o basic-tutorial-1 `pkg-config --cflags --libs gstreamer-1.0`
$ ./basic-tutorial-1 
```

--------------------------------------
하나씩 이해해보도록 하겠습니다. 
```bash
/*init Gstramer*/
gst_init(&args, &argv);
```
이 코드는 항상 첫번째 GStreamer의 command여야 합니다. 
    - 모든 interal structures를 모두 초기화합니다.
    - 어떤 plugin들이 사용가능한지를 확인합니다
    - gstreamer에 의도된 모든 command-line option들을 시랳ㅇ한다. 만약 command-line paramteter argc와 argv를 전달한다면 GStreamer의 기본 command-line option들의 혜택을 얻게 된다. 

```bash
pipeline = 
    gst_parse_launch
    ("playbin uri=https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm",
    NULL);
```
이 라인이 이 튜토리얼의 핵심 코드이다. 여기서 gst_parse_launch와 playbin의 사용법을 이애하게 된다. 

#### gst_parse_launch
Gstreamer는 multimedia flow들을 다루기 위해 디자인되었다. media는 producer인 "source"요소로부터 consumer인 sink까지 여행을 한다. 그리고 이 여행은 모든 종류의 일을 처리하는 중간 요소들을 전달하면서 진행이 된다. 모든 서로 연결된 이 요소들의 집합을 pipeline이라고 부른다. 

이 Gstreamer안에서 우리는 개인 요소들을 모아서 pipeline을 build한다. 

#### playbin
playbin은 source처럼 그리고 sink처럼 행동하는 특별한 요소이면서 완전한 pipeline이다. 내부적으로 media를 실행하는데 필요한 모든 요소들을 생성하고 연결한다. 

```bash
/*start playing*/
get_element_set_state(pipeline, GST_STATE_PLAYING);
```
state를 설정한다. 모든 Gstreamer 요소들을 관련된 state를 가진다. 이를 dvd player에서 존재하는 play/pause button이라고 생각하면 편하다.

```bash
/* Wait until error or EOS */
bus = gst_element_get_bus (pipeline);
msg =
    gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE,
    GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
```
이 라인은 에러가 발생할 때까지 기다리거나 stream이 종료될 때까지 기다린다. 이 함수는 pipeline의 버그를 검색하고 gst_bus_timed_pop_filtered()는 이 버그를 통해서 에러나 EOS를 얻을 때까지 차단된다. 


## 결론 
1. 어떻게 gstreamer를 초기화할 것인가 : gst_init()
2. textual description으로부터 pipeline을 build하는 방법 : gst_parse_launch()
3. 자동으로 재생 pipeline을 만드는 방법: playbin
4. Gstreamer가 실행상태가 되도록 신호를 보내는 방법 : gst_element_set_state()
5. 편안한 작업을 위함 : get_element_get_bus(), get_bus_timed_pop_filtered()

