# Import the OpenCV library
import cv2


# Define a function to detect faces and draw bounding boxes around them

def faceBox(faceNet, frame):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    
    # Preprocess the frame for face detection using a blob

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    
    # Set the blob as input to the face detection model

    faceNet.setInput(blob)
    
    # Perform face detection

    detection = faceNet.forward()
    #print(detection.shape)
    
    # Initialize an empty list to store bounding box coordinates

    bboxs = []
    
    # Loop through the detections

    for i in range(detection.shape[2]):

        # Extract confidence score for the detection

        confidence = detection[0, 0, i, 2]
        
        # If confidence is above a threshold, consider it a valid detection

        if confidence > 0.7:

            # Calculate bounding box coordinates

            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            
            # Append bounding box coordinates to the list

            bboxs.append([x1, y1, x2, y2])
            
            # Draw bounding box around the detected face

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Return the frame with bounding boxes drawn and the list of bounding box coordinates

    return frame, bboxs

# Paths to pre-trained face detection, age estimation, and gender recognition models

faceProto = "C:\\Users\\masta\\Downloads\\AGE & GENDER\\AGE & GENDER\\opencv_face_detector.pbtxt"
faceModel = "C:\\Users\\masta\\Downloads\\AGE & GENDER\\AGE & GENDER\\opencv_face_detector_uint8.pb"
ageProto = "C:\\Users\\masta\\Downloads\\AGE & GENDER\\AGE & GENDER\\age_deploy.prototxt"
ageModel = "C:\\Users\\masta\\Downloads\\AGE & GENDER\\AGE & GENDER\\age_net.caffemodel"
genderProto = "C:\\Users\\masta\\Downloads\\AGE & GENDER\\AGE & GENDER\\gender_deploy.prototxt"
genderModel = "C:\\Users\\masta\\Downloads\\AGE & GENDER\\AGE & GENDER\\gender_net.caffemodel"

# Read pre-trained models using OpenCV

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define mean values for preprocessing and lists of age and gender labels

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-4)', '(4-8)', '(8-12)', '(12-18)', '(18-25)', '(25-50)', '(50-70)', '(70-100)']
genderList = ['Male', 'Female']

# Open video capture from default camera

video = cv2.VideoCapture(0)

# Define padding for face cropping

padding = 20


while True:
    # Read a frame from the video

    ret, frame = video.read()
    
    # Detect faces in the frame and draw bounding boxes

    frame, bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:

        # Crop the face region with padding

        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        
        # Preprocess the face image for gender prediction

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Set the blob as input to the gender recognition model

        genderNet.setInput(blob)
        
        # Perform gender prediction

        genderPred = genderNet.forward()
        
        
        # Get the predicted gender label

        gender = genderList[genderPred[0].argmax()]
        
        # Preprocess the face image for age prediction

        ageNet.setInput(blob)
        
        # Perform age prediction

        agePred = ageNet.forward()
        
        # Get the predicted age label

        age = ageList[agePred[0].argmax()]
        
        # Combine gender and age labels

        label = "{},{}".format(gender, age)
        
        # Draw a filled rectangle as label background

        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        
        # Put the label text on the frame

        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,cv2.LINE_AA)
    
    # Display the frame with gender and age labels

    cv2.imshow("Age-Gender", frame)
    
    # Wait for a key press, break the loop if 'q' is pressed

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

# Release the video capture object and close all OpenCV windows

video.release()
cv2.destroyAllWindows()
