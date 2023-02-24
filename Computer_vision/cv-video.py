def apply_yolo_object_detection(image_to_process):
    """
    Recognition and determination of the coordinates of objects on the image
    :param image_to_process: original image
    :return: image with marked objects and captions to them
    """
height, width, _ = image_to_process.shape
blob = cv2.dnn.blobFromImage(image_to_process, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">1</span> / <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>, (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">608</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">608</span>),
                             (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>), swapRB=<span class="hljs-literal" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">True</span>, crop=<span class="hljs-literal" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">False</span>)
net.setInput(blob)
outs = net.forward(out_layers)
class_indexes, class_scores, boxes = ([] <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">for</span> i <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">in</span> range(<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">3</span>))
objects_count = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>

<span class="hljs-comment" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(142, 144, 140); quotes: &quot;«&quot; &quot;»&quot;;"># Starting a search for objects in an image</span>
<span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">for</span> out <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">in</span> outs:
    <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">for</span> obj <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">in</span> out:
        scores = obj[<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">5</span>:]
        class_index = np.argmax(scores)
        class_score = scores[class_index]
        <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">if</span> class_score &gt; <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>:
            center_x = int(obj[<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>] * width)
            center_y = int(obj[<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">1</span>] * height)
            obj_width = int(obj[<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">2</span>] * width)
            obj_height = int(obj[<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">3</span>] * height)
            box = [center_x - obj_width // <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">2</span>, center_y - obj_height // <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">2</span>,
                   obj_width, obj_height]
            boxes.append(box)
            class_indexes.append(class_index)
            class_scores.append(float(class_score))

<span class="hljs-comment" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(142, 144, 140); quotes: &quot;«&quot; &quot;»&quot;;"># Selection</span>
chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0.0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0.4</span>)
<span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">for</span> box_index <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">in</span> chosen_boxes:
    box_index = box_index
    box = boxes[box_index]
    class_index = class_indexes[box_index]

    <span class="hljs-comment" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(142, 144, 140); quotes: &quot;«&quot; &quot;»&quot;;"># For debugging, we draw objects included in the desired classes</span>
    <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">if</span> classes[class_index] <span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">in</span> classes_to_look_for:
        objects_count += <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">1</span>
        image_to_process = draw_object_bounding_box(image_to_process,
                                                    class_index, box)

final_image = draw_object_count(image_to_process, objects_count)
<span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">return</span> final_image

def draw_object_count(image_to_process, objects_count):
"""
Signature of the number of found objects in the image
:param image_to_process: original image
:param objects_count: the number of objects of the desired class
:return: image with labeled number of found objects
"""
start = (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">10</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">120</span>)
font_size = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">1.5</span>
font = cv2.FONT_HERSHEY_SIMPLEX
width = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">3</span>
text = <span class="hljs-string" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(113, 140, 0); quotes: &quot;«&quot; &quot;»&quot;;">"Objects found: "</span> + str(objects_count)

<span class="hljs-comment" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(142, 144, 140); quotes: &quot;«&quot; &quot;»&quot;;"># Text output with a stroke</span>
<span class="hljs-comment" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(142, 144, 140); quotes: &quot;«&quot; &quot;»&quot;;"># (so that it can be seen in different lighting conditions of the picture)</span>
white_color = (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>)
black_outline_color = (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>)
final_image = cv2.putText(image_to_process, text, start, font, font_size,
                          black_outline_color, width * <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">3</span>, cv2.LINE_AA)
final_image = cv2.putText(final_image, text, start, font, font_size,
                          white_color, width, cv2.LINE_AA)

def draw_object_bounding_box(image_to_process, index, box):
"""
Drawing object borders with captions
:param image_to_process: original image
:param index: index of object class defined with YOLO
:param box: coordinates of the area around the object
:return: image with marked objects
"""
x, y, w, h = box
start = (x, y)
end = (x + w, y + h)
color = (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>)
width = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">2</span>
final_image = cv2.rectangle(image_to_process, start, end, color, width)

start = (x, y - <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">10</span>)
font_size = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">1</span>
font = cv2.FONT_HERSHEY_SIMPLEX
width = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">2</span>
text = classes[index]
final_image = cv2.putText(final_image, text, start, font,
                          font_size, color, width, cv2.LINE_AA)

<span class="hljs-keyword" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; font-weight: 700; color: rgb(137, 89, 168); quotes: &quot;«&quot; &quot;»&quot;;">return</span> final_image
def draw_object_count(image_to_process, objects_count):
"""
Signature of the number of found objects in the image
:param image_to_process: original image
:param objects_count: the number of objects of the desired class
:return: image with labeled number of found objects
"""
start = (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">10</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">120</span>)
font_size = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">1.5</span>
font = cv2.FONT_HERSHEY_SIMPLEX
width = <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">3</span>
text = <span class="hljs-string" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(113, 140, 0); quotes: &quot;«&quot; &quot;»&quot;;">"Objects found: "</span> + str(objects_count)

<span class="hljs-comment" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(142, 144, 140); quotes: &quot;«&quot; &quot;»&quot;;"># Text output with a stroke</span>
<span class="hljs-comment" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(142, 144, 140); quotes: &quot;«&quot; &quot;»&quot;;"># (so that it can be seen in different lighting conditions of the picture)</span>
white_color = (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">255</span>)
black_outline_color = (<span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>, <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">0</span>)
final_image = cv2.putText(image_to_process, text, start, font, font_size,
                          black_outline_color, width * <span class="hljs-number" style="transition: opacity 0.2s ease-in-out 0s, color 0.2s ease-in-out 0s, text-decoration 0.2s ease-in-out 0s, background-color 0.2s ease-in-out 0s, -webkit-text-decoration 0.2s ease-in-out 0s; color: rgb(245, 135, 31); quotes: &quot;«&quot; &quot;»&quot;;">3</span>, cv2.LINE_AA)
final_image = cv2.putText(final_image, text, start, font, font_size,
                          white_color, width, cv2.LINE_AA)

def start_video_object_detection(video: str):
    """
    Real-time video capture and analysis
    """

    while True:
        try:
            # Capturing a picture from a video
            video_camera_capture = cv2.VideoCapture(video)
            
            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()
                if not ret:
                    break
                
                # Application of object recognition methods on a video frame from YOLO
                frame = apply_yolo_object_detection(frame)
                
                # Displaying the processed image on the screen with a reduced window size
                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
                cv2.imshow("Video Capture", frame)
                cv2.waitKey(1)
            
            video_camera_capture.release()
            cv2.destroyAllWindows()
    
        except KeyboardInterrupt:
            pass

def start_video_object_detection(video: str):
    """
    Real-time video capture and analysis
    """

    while True:
        try:
            # Capturing a picture from a video
            video_camera_capture = cv2.VideoCapture(video)
            
            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()
                if not ret:
                    break
                
                # Application of object recognition methods on a video frame from YOLO
                frame = apply_yolo_object_detection(frame)
                
                # Displaying the processed image on the screen with a reduced window size
                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
                cv2.imshow("Video Capture", frame)
                cv2.waitKey(0):
                		break
            
            video_camera_capture.release()
            cv2.destroyAllWindows()
    
        except KeyboardInterrupt:
            pass
        

if __name__ == '__main__':
		# Loading YOLO scales from files and setting up the network
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg",
                                     "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Loading from a file of object classes that YOLO can detect
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    # Determining classes that will be prioritized for search in an image
    # The names are in the file coco.names.txt

    video = input("Path to video (or URL): ")
    look_for = input("What we are looking for: ").split(',')
    
    # Delete spaces
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    start_video_object_detection(video)
    
