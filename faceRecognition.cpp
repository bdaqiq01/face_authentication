#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>   


#include <iostream>
#include <map>
#include <string> 
#include <vector>

using namespace cv;
using namespace cv::face;
using namespace std;

map<int, string> nameDatabase;  // maps inex labels to string names
vector<Mat> images;            // MAt data type from open cv used to represent images or matrices
vector<int> labels;            // each num in labels corresponds to a specific Mat
int currentLabel = 1;          // Incremental label for new faces




Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(); //pointer to a LBPH face recognition model 


void trainModel() {
    if (!images.empty()) {
        model->train(images, labels);
        cout << "Model trained with " << images.size() << " faces." << endl;
    }
}


void addNewPerson(const string& name, const Mat& face) {  //takes in the name and the image
    images.push_back(face);
    labels.push_back(currentLabel);
    nameDatabase[currentLabel] = name;
    currentLabel++;
    trainModel();
}


int main() {
    //the VideoCapture class is used to interface with video stream, e.g camera, video file ...
    // CAP_V4L2 the underlaying backend APU that open cv used to interface with the hardware or software to capture video or images 
    VideoCapture cap(0, cv::CAP_V4L2);  // 0 index/id of the video capturing device to open, 0 default camera to Force V4L2 backend
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera!" << endl;
        return -1;
    }

    CascadeClassifier face_cascade;
    if (!face_cascade.load("../haarcascade_frontalface_default.xml")) {
        cerr << "Error: Could not load Haar cascade file!" << endl;
        return -1;
    }

    cout << "Press 'a' to add a new person to the database." << endl;
    cout << "Press 'q' to quit." << endl;

    while (true) {
        Mat frame, gray; //mat is DT for image in cv, frame for color, gray for the gray scale representatin
        cap >> frame; //captures from the next frame and stores it in the frame
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY); //cvtCOlor converts the color space of an image fram:input, gray:output image, COLOR_BGR2GRAY prefec constrant to convert from BGRTOGRAY
        
        vector<Rect> faces; //stores the bounding rectangles around the face
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30)); //This function detects objects (in this case, faces) in the grayscale image using the Haar cascade classifier (face_cascade).

        for (const auto& face : faces) {
            Mat faceROI = gray(face);
            resize(faceROI, faceROI, Size(100, 100));  // Resize for consistency

            int predictedLabel = -1;
            double confidence = 0.0;
            string labelName = "Unknown";

            // checks if the model is trained then check if u can recognize the face
            if (!images.empty()) {
                model->predict(faceROI, predictedLabel, confidence);
                if (confidence < 80 && nameDatabase.find(predictedLabel) != nameDatabase.end()) {
                    labelName = nameDatabase[predictedLabel];
                }
            }

            // draw rectangle and label
            rectangle(frame, face, Scalar(255, 0, 0), 2);
            putText(frame, labelName, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
        }

        imshow("Face Recognition", frame);

        char key = waitKey(1);
        if (key == 'q') {
            cout << "Exiting program." << endl;
            break;
        } else if (key == 'a') {
            cout << "Enter the name of the person: ";
            string name;
            cin >> name;

            // use the first detected face for adding
            if (!faces.empty()) {
                Mat faceROI = gray(faces[0]);
                resize(faceROI, faceROI, Size(100, 100));
                addNewPerson(name, faceROI);
                cout << name << " added to the database." << endl;
                model->train(images, labels);  // Ensure immediate model update
               
                //faces.clear();
                face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
                for (const auto& face : faces) {
                    Mat faceROI = gray(face);
                    resize(faceROI, faceROI, Size(100, 100));

                    int predictedLabel = -1;
                    double confidence = 0.0;
                    string labelName = "Unknown";

                    if (!images.empty()) {
                        model->predict(faceROI, predictedLabel, confidence);
                        if (confidence < 70 && nameDatabase.find(predictedLabel) != nameDatabase.end()) {
                            labelName = nameDatabase[predictedLabel];
                        }
                    }

                    rectangle(frame, face, Scalar(255, 0, 0), 2);
                    putText(frame, labelName, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
                }
            } else {
                cout << "No face detected. Try again." << endl;
            }
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

