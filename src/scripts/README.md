# MNIST pairwise comparison classification system based on residual networks and mixed-precision training

## Group Members and Divisions

-李松峪：Main Programming, Experiment Analysis, Debugging, Testing, Presentation
-肖骋宇：PPT Preparation
-徐浩博:  Ideal Providing
-李卓霖: Opinion Guidance 
-马鸣禧: Opinion Guidance 
-邓仁杰: Opinion Guidance 
-刘科言: Opinion Guidance 
-薛钰泷: Opinion Guidance 

## Introduction

This is a simple model that inputs a MNIST digit image spliced ​​left and right, and determines whether the left digit is larger than right digit.

This model uses some simple resudial blocks, label swapping, special feature comparison to improve its study. 

## Environmental Requirements

### Hardware Requirement

Operating System: Linux/Windows/MacOS

Minimun RAM: 4GB

Minimun Disk Storage: 4GB

GPU Requirement: Not Needed

### Software Requirement

Python >=3.8
torch>=2.2
torchvision>=0.17
numpy>=1.23
pandas>=1.5
tqdm>=4.65
matplotlib>=3.5
seaborn>=0.12
scikit-learn>=1.2

Except Python, all of the other relying configuration can be downloaded via requirements.txt


## Performance

### Expirement Hardware Environment

Cloud Platform: Google Cloud Platform

Configuration name: E2-highcpu-16

RAM: 16GB

CPU: Intel Broadwell 16 cores

Disk Storage: 64GB

GPU: Not used

### Outcomes

-model param number: 504,258

-Training Time: 7 mins and 40 secs

-Accuracy on public test : About 0.76 (precisely 0.759)

-Accuracy on validation set: About 0.74 (precisely 0.7401)

## How to Run

**We provide a simple shell script in order to run it simply**

**Changing "train_baseline.txt" to "train_baseline.py if you want learning curves**

**The Seed will be set up to 42 automatically for we do not need to change the seed**

**Before Running this program,drag the data folder to ./src**

Thus, we can run this program on these steps:

1-Go to this project: cd /path/to/this project/src

2-Create a virtual machine: python3 -m venv .venv

3-Enter virtual machine:source .venv/bin/activate

4-Install dependency: cd ./scripts && pip install -r requirements.txt

5-Giving the shell permissions to execute: chmod +x start.sh

6- Start the shell: cd .. && ./scripts/start.sh

7- Wait for the outcome，the outcome will generated on ./outputs directory


## Project Structure

📦src
 ┣ 📂scripts
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜baseline_inference.cpython-311.pyc
 ┃ ┃ ┣ 📜baseline_inference.cpython-313.pyc
 ┃ ┃ ┣ 📜check_submission.cpython-311.pyc
 ┃ ┃ ┣ 📜check_submission.cpython-313.pyc
 ┃ ┃ ┣ 📜eval_public.cpython-311.pyc
 ┃ ┃ ┣ 📜eval_public.cpython-313.pyc
 ┃ ┃ ┣ 📜ta_make_dataset_corrupted.cpython-311.pyc
 ┃ ┃ ┣ 📜train_baseline.cpython-311.pyc
 ┃ ┃ ┣ 📜train_baseline.cpython-312.pyc
 ┃ ┃ ┣ 📜train_baseline.cpython-313.pyc
 ┃ ┃ ┗ 📜train_baseline.cpython-38.pyc
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜simple_compare_cnn.cpython-311.pyc
 ┃ ┃ ┃ ┣ 📜simple_compare_cnn.cpython-313.pyc
 ┃ ┃ ┃ ┗ 📜simple_compare_cnn.cpython-38.pyc
 ┃ ┃ ┗ 📜simple_compare_cnn.py
 ┃ ┣ 📂outputs
 ┃ ┣ 📂utils
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜corruptions.cpython-311.pyc
 ┃ ┃ ┃ ┣ 📜data.cpython-311.pyc
 ┃ ┃ ┃ ┣ 📜data.cpython-313.pyc
 ┃ ┃ ┃ ┣ 📜data.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜metrics.cpython-311.pyc
 ┃ ┃ ┃ ┣ 📜metrics.cpython-313.pyc
 ┃ ┃ ┃ ┣ 📜seed.cpython-311.pyc
 ┃ ┃ ┃ ┣ 📜seed.cpython-313.pyc
 ┃ ┃ ┃ ┗ 📜seed.cpython-38.pyc
 ┃ ┃ ┣ 📜corruptions.py
 ┃ ┃ ┣ 📜data.py
 ┃ ┃ ┣ 📜metrics.py
 ┃ ┃ ┗ 📜seed.py
 ┃ ┣ 📜.DS_Store
 ┃ ┣ 📜README.md
 ┃ ┣ 📜baseline_inference.py
 ┃ ┣ 📜check_submission.py
 ┃ ┣ 📜eval_public.py
 ┃ ┣ 📜requirements.txt
 ┃ ┣ 📜start.sh
 ┃ ┣ 📜test.py
 ┃ ┣ 📜train_baseline.py
 ┃ ┗ 📜train_baseline.txt
 ┗ 📜.DS_Store

 📜pred_private.csv

 📜requirements.txt

 📜ML_Group4_presentation.pptx

 📜README.md