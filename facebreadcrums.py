import cv2
import numpy

def get_intensity_callback(event,x,y,flags,image):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x)
        print(y)
        print(image[y,x])
        print("---")

def create_breadcrumb_map(faces, height, width):
    breadcrumby, breadcrumbx = numpy.mgrid[0:height, 0:width]
    breadcrumb = numpy.zeros((height, width), numpy.float)
    for (x_face, y_face, width_face, height_face) in faces:
        x_face_centre = x_face+width_face/2
        y_face_centre = y_face+height_face/2
        breadcrumb -= numpy.sqrt(numpy.maximum(abs(breadcrumbx - x_face_centre) - width_face / 2, 0) ** 2 + numpy.maximum(abs(breadcrumby - y_face_centre) - height_face / 2, 0)**2).astype(float)
    breadcrumb -= numpy.min(breadcrumb)
    breadcrumb /= numpy.max(breadcrumb)
    return breadcrumb

def get_breadcrumb_map_to_faces(imageLocation):
    img = cv2.imread(imageLocation, 0)
    face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    height, width = img.shape
    return create_breadcrumb_map(faces, height, width)

breadcrumb = get_breadcrumb_map_to_faces('resources/tennis-b.jpg')
breadcrumb *= 255

cv2.imshow('breadcrumb', breadcrumb)
cv2.setMouseCallback('breadcrumb', get_intensity_callback, breadcrumb)
cv2.imwrite('resources/facebreadcrumb.jpg',breadcrumb)
cv2.waitKey(0)
cv2.destroyAllWindows()
