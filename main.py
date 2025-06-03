import face_recognition
import os
import cv2

dataset = "dataset"
test = "Test Data"
tolerance = 0.6 #similar to temperature, lower the better the recognition but takes more time the lower it is
frame_thickness = 3 #square around head / number of pixels we are using
font_thickness = 2
MODEL = "cnn"

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(dataset):
    for filename in os.listdir(f"{dataset}\{name}"):
        image = face_recognition.load_image_file(f"{dataset}\{name}\{filename}")
        encodings = face_recognition.face_encodings(image) # here we are specifying zero as we only need one main identity per image not all
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(name)
        else:
            print(f"No face found in {filename}")

print("processing unknown faces")

for filename in os.listdir(test):
    print(filename)
    image = face_recognition.load_image_file(f"{test}\{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    if not encodings:
        print(f"No faces found in {filename}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces,face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1],face_location[2])
            color = [0,255,0]
            cv2.rectangle(image, top_left,bottom_right, color , frame_thickness)

            top_left = (face_location[3],face_location[2])
            bottom_right = (face_location[1],face_location[2]+22)
            cv2.rectangle(image, top_left,bottom_right, color , cv2.FILLED)
            cv2.putText(image,match,(face_location[3],face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),font_thickness)
        else:
            print(f"No face found in {filename}")

    cv2.imshow(filename,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





