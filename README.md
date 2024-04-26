# Key_detect
As the name suggests the main goal if this project is detecting keys on a keyboard of a mobile phone. Such feature was developed mainly for an autonomous robotic arm (MOBOT) capable of setting up mobile phones for in Škoda factory. The Key_detect employs YOLOv5 network, which is then trained on a custom dataset containing keyboard images. The dataset contained various keyboard variants coming in a form of color and types of keyboards on different brands of phones. The training dataset was augmented in order to achieve more robust predictions. An example of the models prediction is present in the following image.
![Example of the model prediction](/presentation/fig/detect40.png)


Finally, as the model is unable to achieve 100% accuracy, the project also contains a postprocessing algorithm, which is able to correct the predictions of the model. The postprocessing algorithm is based on the assumption that the keys are placed in a grid and that the keys are not overlapping. The algorithm is able to correct the predictions of the model by removing overlapping keys and by adding missing keys. The following image shows the ability to add missing keys even in the case of a glare.
![Prediction with the correction](/presentation/fig/completedGlare.png)

## Usage
First download the needed libraries from the requirements.txt file.
```bash
pip install -r requirements.txt
```
To detect the keys on the keyboard, run the following command.
```bash
python customDetect.py --path path_to_image
```
Where `--path` is an optional argument, alternatively an example file is used for the detection. Currently, the software just displays the predictions on the users screen.

### Credits
The project was created as a volountary work during VIR subject at the CTU FEE and helped to achieve excelent grade. We would like to thank the teachers of the subject for their help and guidance. Additionally, we followed the [sciencedirect tutorial](https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9), which helped us when working with the YOLOv5 network.