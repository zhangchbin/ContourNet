### ContourNet
Richer Convolutional Features for Contour Detection on PASCAL VOC 2012 Dataset.

### Introduction
My ContourNet(CNet) can be used to detection the object contour on PASCAL VOC 2012 Dataset.
CNet refer <a href="http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Richer_Convolutional_Features_CVPR_2017_paper.pdf">RCF</a>,
<a href="http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Learning_a_Discriminative_CVPR_2018_paper.pdf">DFN</a> and <a href="http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Object_Contour_Detection_CVPR_2016_paper.pdf">CEDN</a>.

### Dataset and Preprocessed
In <a href="http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Object_Contour_Detection_CVPR_2016_paper.pdf">CEDN</a>, they proposed the method of preprocessing data. Actually, they have provided the <a href="https://github.com/jimeiyang/objectContourDetector/blob/master/data/PASCAL/get_pascal_training_data.sh">preprocessed-data</a>.

### My Results
| Method |Original Motivation|ODS F-score|OIS F-score|AP|
|:---|:---|:---|:---|:---|
| CNet(Ours) |Contour Detection|0.404|0.509|0.373|
| CEDN |Contour Detection|0.486(in the same environment),0.57(in CEDN paper)|0.5(in the same environment), -|0.354(in the same environment), -|
| RCF |Edge Detection|0.459(in my implement and training)|0.475(in my implement and training)|0.333(in my implement and training)|
| HED |Edge Detection|0.441(in my implement and training)|0.454(in my implement and training)|0.311(in my implement and training)|

### Environment
pytorch-1.0.0, numpy, PIL

### Thanks
Thanks to <a href="https://github.com/meteorshowers">XuanYi Li</a>, <a href="https://github.com/shenwei1231">Wei Shen</a>.
