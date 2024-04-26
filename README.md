# Key_detect
As the name suggests, the main goal of this project is detecting keys on the keyboard of a mobile phone. Such a feature was developed mainly for an autonomous robotic arm (MOBOT) capable of setting up mobile phones in the Škoda factory. The Key_detect employs the YOLOv5 network and is trained on a custom keyboard image dataset. The dataset contained various keyboard variants coming in color and types of keyboards on different brands of phones. The training dataset was augmented to achieve more robust predictions. An example of the model's prediction is presented in the following image.
![Example of the model prediction](/presentation/fig/detect40.png)


Finally, as the model cannot achieve 100% accuracy, the project also contains a postprocessing algorithm, which can correct the predictions of the model. The postprocessing algorithm assumes that the keys are placed in a grid and that the keys are not overlapping. The algorithm can correct the model's predictions by removing overlapping keys and adding missing ones. The following image shows the ability to add missing keys even in the case of a glare.
![Prediction with the correction](/presentation/fig/completedGlare.png)

## Usage
First, download the needed libraries from the requirements.txt file.
```bash
pip install -r requirements.txt
```
To detect the keys on the keyboard, run the following command.
```bash
python customDetect.py --path path_to_image
```
Where `--path` is an optional argument. Alternatively, an example file is used for the detection. Currently, the software displays the predictions on the user's screen.

### Credits
The project was created as a voluntary work during the VIR subject at the CTU FEE and helped to achieve excellent grades. We want to thank the teachers of the subject for their help and guidance. Additionally, we followed the [sciencedirect tutorial](https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9), which helped us when working with the YOLOv5 network. Additionally, [Roboflow](https://roboflow.com/) augmented the dataset.