# PCPO 项目数据集下载链接整理

本项目实验共涉及 4 个域适应（Domain Adaptation）数据集。由于部分早期数据集的官方服务器经常不稳定或已失效，以下除了提供官方链接外，也提供了 Kaggle 等高可用替代下载方案，确保每个数据集都能顺利获取。

## 域适应 (Domain Adaptation) 数据集

1. **Office-31**
   - **官方链接/主页**: [网页链接](https://faculty.cc.gatech.edu/~judy/domainadapt/) / [备份主页](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/) (常连接超时)
   - **高可用替代方案 (Kaggle)**: [https://www.kaggle.com/datasets/tanlikesmath/office31](https://www.kaggle.com/datasets/tanlikesmath/office31)
   
2. **Office-Home**
   - **官方主页**: [http://hemanthdv.org/OfficeHome-Dataset/](http://hemanthdv.org/OfficeHome-Dataset/)
   - **说明**: 网页正常，通常需要在官网填写表单或直接使用官网内关联 of Google Drive/OneDrive 获取。
   - **高可用替代方案 (Kaggle)**: [https://www.kaggle.com/datasets/jessicali9530/officehome-dataset](https://www.kaggle.com/datasets/jessicali9530/officehome-dataset)

3. **VisDA-2017**
   - **官方主页**: [http://ai.bu.edu/visda-2017/](http://ai.bu.edu/visda-2017/)
   - **官方 GitHub**: [https://github.com/VisionLearningGroup/taskcv-2017-public](https://github.com/VisionLearningGroup/taskcv-2017-public)
   - **说明点**: 包含 Train (合成数据集) 和 Validation/Test (真实图像)，可直接在 GitHub 的说明页面和[基准测试脚本里获取直链](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)。

4. **DomainNet (包含 miniDomainNet)**
   - **官方主页**: [http://ai.bu.edu/M3SDA/](http://ai.bu.edu/M3SDA/)
   - **说明**: 直接在页面中可下载 clipart、painting、real、sketch 等 6 个域的内容 (请下载 "cleaned version" 分片)。
   - **miniDomainNet Note**: 本项目中也使用到了 mini 版，基于 DomainNet 切分，其对应的官方切分文件由作者托管在 Google Drive: [点击这里下载](https://drive.google.com/open?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7)
