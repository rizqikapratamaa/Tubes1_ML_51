# 𝙵𝚎𝚎𝚍𝚏𝚘𝚛𝚠𝚊𝚛𝚍 𝙽𝚎𝚞𝚛𝚊𝚕 𝙽𝚎𝚝𝚠𝚘𝚛𝚔

> **Tugas Besar IF3270 - Machine Learning**

> A Feedforward Neural Network (FFNN) is a type of artificial neural network that consists of multiple layers of neurons connected in a one-way fashion, without feedback loops. FFNNs are widely used in various machine learning applications such as classification, regression, and pattern recognition. The model processes input data step by step through hidden layers before producing a final output.

<img src="src" alt="main-image-here" width="800"/>

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setting Up](#setting-up)
4. [Bonus](#bonus-🤑)
5. [Acknowledgements](#acknowledgements)

## Overview

This project focuses on implementing a **Feedforward Neural Network (FFNN)** from scratch, without using deep learning libraries such as TensorFlow or PyTorch. The developed model supports various customizable parameters, including:

   - Number of layers and number of neurons per layer
   - Activation function for each layer
   - Loss function used in the model
   - Weight initialization methods
   - Forward propagation and backward propagation process
   - Weight updates using gradient descent

In this implementation:
   - **Hidden layers** use the `Sigmoid` activation function
   - **The output layer** uses the `Softmax` activation function for multi-class classification tasks

The project report will also include experiments on different hyperparameters, analyzing the impact of depth & width, activation functions, learning rate, weight initialization, and a comparison with sklearn’s MLPClassifier.

## Important Notice
>
> [!IMPORTANT]\
> This project requires several Python libraries. Make sure to install all dependencies before running the code. See the [Requirements](#requirements) section for details.

## Requirements

Programming Language: Python
Libraries Used:
   - `numpy==2.2.3` for numerical computations
   - `matplotlib` for visualization
   - `scikit-learn==1.6.1` for comparison with MLPClassifier
   - `scipy==1.15.2` for scientific computing
   - `joblib==1.4.2` for model persistence
   - `threadpoolctl==3.5.0` for managing thread pools

## Setting Up

>
> [!NOTE]\
> This setup is for WSL. If you are developing on Windows, create your own virtual environment.

<details>
<summary>:eyes: Get Started</summary>
#### Clone the Repository:

```sh
 git clone https://github.com/rizqikapratamaa/Tubes1_ML_51.git
 cd TUBES1_ML_51
```

#### Create new env

```sh
 python3 -m venv env_tubes
 source env_tubes/bin/activate
```

#### Install requirements

```sh
 pip install -r requirement.txt
```

#### Run the program

```sh
 python3 main.py
```

#### After finishing, exit from venv

```sh
 python3 main.py
```
</details>

## Bonus 🤑
<details>
<summary>1. Other activation functions</summary>

Details

> Responsible: 13522147
</details>

<details>
<summary>2. Other init methods</summary>

Details

> Responsible: 13522139
</details>

<details>
<summary>3. L1 L2 Regulation Method</summary>

Details

> Responsible: 13522126 
</details>

<details>
<summary>4. RMSNorm </summary>

Details

> Responsible: 13522126 
</details>

## Screenshot

<details>
<summary>📸 Open me</summary>

<img src="src" alt="example-image-here" width="800"/>

</details>

## Connect

| Name                      | NIM      | Connect                                                |
| ------------------------- | -------- | ------------------------------------------------------ |
| Rizqika Mulia Pratama | 13522126 | [@rizqikapratamaa](https://github.com/rizqikapratamaa) |
| Attara Majesta Ayub       | 13522139 | [@attaramajesta](https://github.com/attaramajesta)     |
| Ikhwan Al Hakim | 13522147 | [@Nerggg](https://github.com/Nerggg) |

## Acknowledgements

- Machine Learning Course Lecturer, Bandung Institute of Technology, 2025
- Machine Learning Teaching Assistants, Bandung Institute of Technology, 2025
