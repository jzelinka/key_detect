We want to train a yolo network to detect our keyboard characters.
First test is conducted using [sciencedirect tutorial](https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9).
It is necessary to download [weights](https://pjreddie.com/media/files/yolov3.weights) to use the YOLO model and place them into config dir.
For faster learning change the value of batch size in `config/yolov3.cfg`.