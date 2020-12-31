## goal
파이프라인을 즉석에서 building하는 방법에 대해 공부한다. 

## introduction
파이프라인자체는 playing state가 되기 전에는 완벽하게 `build`된 상태라고 보기 어렵다. 
이 예제에서 우리는 container file에서 media file인 오디오와 비디오가 함께 저장된 파일을 열어보고자 한다. 
이런 `muxer`는 Mastroska, Quick Time, Ogg or Advantage format과 같은 형태를 가진다. 
<br/>
예를 들어 컨테이너가 multiple streams를 embed하고 있다면 demuxer는 이들을 분리하고 각각을 서로 다른 output port로 출력하게 한다. 이 과정에서 다른 branch들이 pipeline안에서 다른 타입으로 생성이 가능하다.
<br/>
Gstreamer요소들이 커뮤니케이션하는데 사용되는 port를 pads라고 하고, 데이터가 요소로 들어가는데 필요한 sink pads와 데이터가 요소로 나오는 source pads가 있다. 
![image](https://user-images.githubusercontent.com/49298791/90365627-05ce8b00-e0a1-11ea-95a6-80f064880a6b.png)

하나의 demuxer는 mixted된 데이터가 들어오는 sink pad와 컨테이너에서 찾아진 stream 각각을 위한 src pads를 가진다. 
![image](https://user-images.githubusercontent.com/49298791/90365783-56de7f00-e0a1-11ea-9555-9ae3b200deb4.png)

아래의 그림을 그 예시가 되는 pipeline에 대한 그림입니다. 
![image](https://user-images.githubusercontent.com/49298791/90365842-71185d00-e0a1-11ea-8efc-f2fb5ef17734.png)

demxer을 다루는 복잡함은 demxer가 데이터를 받아 컨테이너 안에 뭐가 들어있는지 보기전엔 어떤것도 생산이 불가능하다는 점에 있다. demxer는 반드시 src pads로 시작하는 형태가 아니며, sink pads로 시작한다. 따라서 demxer의 src pads에서 출력되는 데이터들을 반드시 terminate시킬 필요성이 생긴다. 

이런 문제를 해결하는 방법은 pipeline을 source에서 demxer로 build하는 것이다. 만약 demxer가 충분한 정보를 바당 어떤 종류, 얼마나 많은 stream이 있는지 알게 되면 demxers는 source pads를 생성하고 이 과정이 pipeline build를 마치는 순간이다. pipeline을 새롭게 생성한 demxer의 src pads에 붙여야 하고 이런 특성에 의해 dynamic pipeline이 된다. 


## dynamic hello word
```c
//basic-tutorial-3.c
#include <gst/gst.h>

/* Structure to contain all our information, so we can pass it to callbacks */
typedef struct _CustomData {
  GstElement *pipeline;
  GstElement *source;
  GstElement *convert;
  GstElement *sink;
} CustomData;

/* Handler for the pad-added signal */
static void pad_added_handler (GstElement *src, GstPad *pad, CustomData *data);

int main(int argc, char *argv[]) {
  CustomData data;
  GstBus *bus;
  GstMessage *msg;
  GstStateChangeReturn ret;
  gboolean terminate = FALSE;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  /* Create the elements */
  data.source = gst_element_factory_make ("uridecodebin", "source");
  data.convert = gst_element_factory_make ("audioconvert", "convert");
  data.sink = gst_element_factory_make ("autoaudiosink", "sink");

  /* Create the empty pipeline */
  data.pipeline = gst_pipeline_new ("test-pipeline");

  if (!data.pipeline || !data.source || !data.convert || !data.sink) {
    g_printerr ("Not all elements could be created.\n");
    return -1;
  }

  /* Build the pipeline. Note that we are NOT linking the source at this
   * point. We will do it later. */
  gst_bin_add_many (GST_BIN (data.pipeline), data.source, data.convert , data.sink, NULL);
  if (!gst_element_link (data.convert, data.sink)) {
    g_printerr ("Elements could not be linked.\n");
    gst_object_unref (data.pipeline);
    return -1;
  }

  /* Set the URI to play */
  g_object_set (data.source, "uri", "https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm", NULL);

  /* Connect to the pad-added signal */
  g_signal_connect (data.source, "pad-added", G_CALLBACK (pad_added_handler), &data);

  /* Start playing */
  ret = gst_element_set_state (data.pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr ("Unable to set the pipeline to the playing state.\n");
    gst_object_unref (data.pipeline);
    return -1;
  }

  /* Listen to the bus */
  bus = gst_element_get_bus (data.pipeline);
  do {
    msg = gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE,
        GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

    /* Parse message */
    if (msg != NULL) {
      GError *err;
      gchar *debug_info;

      switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_ERROR:
          gst_message_parse_error (msg, &err, &debug_info);
          g_printerr ("Error received from element %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
          g_printerr ("Debugging information: %s\n", debug_info ? debug_info : "none");
          g_clear_error (&err);
          g_free (debug_info);
          terminate = TRUE;
          break;
        case GST_MESSAGE_EOS:
          g_print ("End-Of-Stream reached.\n");
          terminate = TRUE;
          break;
        case GST_MESSAGE_STATE_CHANGED:
          /* We are only interested in state-changed messages from the pipeline */
          if (GST_MESSAGE_SRC (msg) == GST_OBJECT (data.pipeline)) {
            GstState old_state, new_state, pending_state;
            gst_message_parse_state_changed (msg, &old_state, &new_state, &pending_state);
            g_print ("Pipeline state changed from %s to %s:\n",
                gst_element_state_get_name (old_state), gst_element_state_get_name (new_state));
          }
          break;
        default:
          /* We should not reach here */
          g_printerr ("Unexpected message received.\n");
          break;
      }
      gst_message_unref (msg);
    }
  } while (!terminate);

  /* Free resources */
  gst_object_unref (bus);
  gst_element_set_state (data.pipeline, GST_STATE_NULL);
  gst_object_unref (data.pipeline);
  return 0;
}

/* This function will be called by the pad-added signal */
static void pad_added_handler (GstElement *src, GstPad *new_pad, CustomData *data) {
  GstPad *sink_pad = gst_element_get_static_pad (data->convert, "sink");
  GstPadLinkReturn ret;
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;

  g_print ("Received new pad '%s' from '%s':\n", GST_PAD_NAME (new_pad), GST_ELEMENT_NAME (src));

  /* If our converter is already linked, we have nothing to do here */
  if (gst_pad_is_linked (sink_pad)) {
    g_print ("We are already linked. Ignoring.\n");
    goto exit;
  }

  /* Check the new pad's type */
  new_pad_caps = gst_pad_get_current_caps (new_pad);
  new_pad_struct = gst_caps_get_structure (new_pad_caps, 0);
  new_pad_type = gst_structure_get_name (new_pad_struct);
  if (!g_str_has_prefix (new_pad_type, "audio/x-raw")) {
    g_print ("It has type '%s' which is not raw audio. Ignoring.\n", new_pad_type);
    goto exit;
  }

  /* Attempt the link */
  ret = gst_pad_link (new_pad, sink_pad);
  if (GST_PAD_LINK_FAILED (ret)) {
    g_print ("Type is '%s' but link failed.\n", new_pad_type);
  } else {
    g_print ("Link succeeded (type '%s').\n", new_pad_type);
  }

exit:
  /* Unreference the new pad's caps, if we got them */
  if (new_pad_caps != NULL)
    gst_caps_unref (new_pad_caps);

  /* Unreference the sink pad */
  gst_object_unref (sink_pad);
}
```



지금까지지는 우리가 필요한 정보를 지역변수로 처리하여 함수 내에서 처리했는데 여기서는 callback 함수가 사용되기 때문에 구조체를 만들어 다루도록 한다. 
```c
/* Structure to contain all our information, so we can pass it to callbacks */
typedef struct _CustomData {
  GstElement *pipeline;
  GstElement *source;
  GstElement *convert;
  GstElement *sink;
} CustomData;
```

pad_added_hander()가 나중에 사용될 것임을 명시해 주고 있는 형태이다. 
```c
/*Handler for the pad-added signal */
static void pad_added_handler (GstElement *src, GstPad *pad, CustomData *data);
```

아래의 코드는 각각의 요소들의 생성해주는 형태이다. 
```c
/* Create the elements */
data.source = gst_element_factory_make ("uridecodebin", "source");
data.convert = gst_element_factory_make ("audioconvert", "convert");
data.sink = gst_element_factory_make ("autoaudiosink", "sink");
```

uridecodebin는 내부적으로 필요한 sources, demuxers and decoders를 instant로 만들어 사용한다. 즉 이는 raw audio와 raw video stream으로 전환하기 위해 필요한 작업니다. 따라서 uridecodebin은 온전한 기능을 하는 playbin에 비교하여 절반의 역할을 해주는 형태이다. 왜냐하면 demuxer의 sourcepads는 처음엔 사용될 수 없고 나중에 즉석으로 link되어야 하기 때문이다. 

