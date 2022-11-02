git clone --recurse-submodules https://github.com/ShoaibSajid/Yolov5_DeepSORT_PseudoLabels.git
cd Yolov5_DeepSORT_PseudoLabels 
pip install -r requirements.txt
rm -r yolov5
git clone --recursive https://github.com/ShoaibSajid/yolov5.git

sudo apt-get install libfreetype6-dev -y
pip uninstall pillow
pip install --no-cache-dir pillow