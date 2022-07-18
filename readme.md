## Evaluating 
---
### Installation
```
    pytorch
    pip3 install IQA-pytorch
```
### Prepare data for evaluation
```
    ./OriginalFolder (High Resolution Folder)
        img1.jpg
        img2.jpg
        ...
    ./EnhancedFolder
        img1.jpg
        img2.jpg
        ... 
```
### Run evaluating PSNR, SSIM
```
    python3 evaluate_PNSR_SSIM.py --original_folder ./OriginalFolder --enhanced_folder ./EnhancedFolder
```