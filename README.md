# Natias
This repository contains the code implementation of the paper "Natias: Neuron Attribution based Transferable Image Adversarial Steganography".


## Abstract
Image steganography is a technique to conceal secret messages within digital images. Steganalysis, on the contrary, aims to detect the presence of secret messages within images.
Recently, deep-learning-based steganalysis methods have achieved excellent detection performance.
As a countermeasure, adversarial steganography has garnered considerable attention due to its ability to effectively deceive deep-learning-based steganalysis. However, steganalysts often employ unknown steganalytic models for detection. Therefore, the ability of adversarial steganography to deceive non-target steganalytic models, known as transferability, becomes especially important. Nevertheless, existing adversarial steganographic methods do not consider how to enhance transferability. To address this issue, we propose a novel adversarial steganographic scheme named Natias. Specifically, we first attribute the output of a steganalytic model to each neuron in the target middle layer to identify critical features. Next, we corrupt these critical features that may be adopted by diverse steganalytic models. Consequently, it can promote the transferability of adversarial steganography. Our proposed method can be seamlessly integrated with existing adversarial steganography frameworks.
Thorough experimental analyses affirm that our proposed technique possesses improved transferability when contrasted with former approaches, and it attains heightened security in retraining scenarios.

## Run
```bash
python split_gen_adv_stego.py --train_cover_dir train_cover_dir --val_cover_dir val_cover_dir --test_cover_dir test_cover_dir --train_rho_dir train_rho_dir --val_rho_dir val_rho_dir --test_rho_dir test_rho_dir --adv_stego_dir adv_stego_dir --batch_size batch_size --model model --ckpt_dir ckpt_dir --payload payload
```

## Contact
If you have any questions, please contact: fanzx@mail.ustc.edu.cn

# Citing

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@ARTICLE{10580936,
  author={Fan, Zexin and Chen, Kejiang and Zeng, Kai and Zhang, Jiansong and Zhang, Weiming and Yu, Nenghai},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Natias: Neuron Attribution-Based Transferable Image Adversarial Steganography}, 
  year={2024},
  volume={19},
  number={},
  pages={6636-6649},
  keywords={Steganography;Distortion;Feature extraction;Security;Deep learning;Convolutional neural networks;Source coding;Adversarial examples;transferability;attribution of deep networks;image steganography;steganalysis},
  doi={10.1109/TIFS.2024.3421893}
}
```
