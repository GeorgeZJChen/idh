import React, { Component } from 'react'
import '../css/content.css'
import Figure from './figure.js'
import Windows from './window.js'
import MyTable from './table.js'
import {Cite, References} from './cite.js'
import MathJax from './math.js'


class Content extends Component {
  constructor(props){
    super(props)
    this.titles = []
    this.references = [
      {
        name: 'Nature',
        text: 'S. Bisdas, H. Shen, S. Thust, V. Katsaros, G. Stranjalis, C. Boskos, S. Brandner and J. Zhang, “Texture analysis-and support vector machine-assisted diffusional kurtosis imaging may allow in vivo gliomas grading and IDH-mutation status prediction: a preliminary study,” Scientific reports, vol. 8, no. 1, p. 6108, 2018.'
      },
      {
        name: '2',
        text: 'A. Tietze, M. B. Hansen, L. Østergaard, S. N. Jespersen, R. Sangill, T. E. Lund, M. Geneser, M. Hjelm and B. Hansen, “Mean diffusional kurtosis in patients with glioma: initial results with a fast imaging method in a clinical setting,” American Journal of Neuroradiology, 2015.'
      },
      {
        name: '3',
        text: 'M. Varma and A. Zisserman, “A statistical approach to texture classification from single images,” International journal of computer vision, vol. 62, pp. 61-68, 2005. '
      },
      {
        name: '4',
        text: 'D. W. McRobbie, E. A. Moore, M. J. Graves and M. R. Prince, MRI from Picture to Proton, Cambridge university press, 2017.'
      },
      {
        name: '5',
        text: 'B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby and Y. Burren, “The multimodal brain tumor image segmentation benchmark (BRATS),” IEEE transactions on medical imaging, vol. 34, no. 10, p. 1993, 2015. '
      },
      {
        name: '6',
        text: 'M. Kistler, S. Bonaretti, M. Pfahrer, R. Niklaus and P. Büchler, “The virtual skeleton database: an open access repository for biomedical research and collaboration,” Journal of medical Internet research, vol. 15, no. 11, 2013. '
      },
      {
        name: '7',
        text: 'S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” arXiv preprint arXiv:1502.03167, 2015. '
      },
      {
        name: '8',
        text: 'G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever and R. R. Salakhutdinov, “Improving neural networks by preventing co-adaptation of feature detectors,” arXiv preprint arXiv:1207.0580 , 2012.'
      },
      {
        name: '9',
        text: 'A. Ng, “Support Vector Machines | Large Margin Intuition,” 2016. [Online]. Available: https://www.youtube.com/watch?v=Ccje1EzrXBU&index=71&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN. [Accessed 1 8 2018].'
      },
      {
        name: '10',
        text: 'J. J. Lee, B. Knox, J. Baumann, C. Breazeal and D. DeSteno, “Computationally modeling interpersonal trust,” Frontiers in psychology, vol. 4, p. 893, 2013.'
      },
      {
        name: '11',
        text: 'K. Simonyan and A. Zisserman., “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014. '
      },
      {
        name: '12',
        text: 'C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov and A. Rabinovich, “Going deeper with convolutions,” In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9, 2015.'
      },
      {
        name: '13',
        text: 'C. Szegedy, S. Ioffe, V. Vanhoucke and A. A. Alemi, “Inception-v4, inception-resnet and the impact of residual connections on learning,” AAAI, vol. 4, p. 12, 2017.'
      },
      {
        name: '14',
        text: 'K. He, X. Zhang, S. Ren and J. Sun, “Deep Residual Learning for Image Recognition,” eprint arXiv:1512.03385, 2015.'
      },
      {
        name: '15',
        text: 'C.-C. Chang and C.-J. Lin, “LIBSVM: a library for support vector machines,” ACM transactions on intelligent systems and technology (TIST), vol. 2, no. 3, p. 27, 2011.'
      }
    ]
    this.Cite = this.Cite.bind(this)
    this.SectionTitle = this.SectionTitle.bind(this)
    this.Subtitle = this.Subtitle.bind(this)
  }
  componentDidMount(){
    this.props.parent.refs.title.update(this.titles)
  }
  Cite(props){
    return <Cite name={props.name} references={this.references} inline={props.inline}/>
  }
  SectionTitle(props){
    this.titles.push({
      level: 1,
      text: props.title
    })
    return <React.Fragment>
    <h2 className='sec-title'>{props.title}<a name={props.title} className='anchor'>{props.title}</a></h2>
    <hr className='sec-s'/>
    </React.Fragment>
  }
  Subtitle(props){
    this.titles.push({
      level: props.level?props.level:2,
      text: props.title
    })
    return <React.Fragment>
    <h3 className='sec-sub-title'>{props.title}<a name={props.title} className='anchor'>{props.title}</a></h3>
    <hr className='sec-s-s'/>
    </React.Fragment>
  }
  render() {
    return (
      <div className="content">

        <hr/>
        <this.SectionTitle title='Keywords'/>
        <p>Machine Learning, Neural Network, Deep Learning, SVM, MRI, 3D Convolution,
        IDH-mutation, Brain Tumour</p>

        <this.SectionTitle title='Abstract'/>
        <p>Some evidence suggests that first-order statistics biomarkers in feature extracted brain tumour images help IDH-mutation status prediction. MR8 filters and 2-D convolution filters are found competent of making prediction on FLAIR and, especially on DKI data. Following the previous works, this paper sets forth the extension of experiments to 3-D convolution, continuing on extracting biomarkers on the 3-D level. Varied kinds of methods were applied. Traditional and novel machine learning techniques were practiced. By utilising 3-D convolutional neural network and support vector machine collaboratively, the accuracies for IDH-mutation status prediction achieved 86.49%, on both FLAIR and DKI modalities. This result supplements the previous findings on this subject.</p>

        <this.SectionTitle title='Introduction'/>

        <p>Magnetic resonance imaging (MRI) plays an important role in providing accurate and efficient diagnosis as it forms anatomy and the physiological processes of tissues and thus helps to distinguish them with clear contrasts between different tissues, either pathological or healthy. By empirical approaches, lesions or normal tissues of the body can be easily demarcated by experienced medical professionals by virtue of high resolution MRI.
          As articulated in <this.Cite name='Nature'/>, diffusion-weighted (DWI) MRI, which is the imaging technique specifically designed for some organs including brain tumour diagnosis, stands on the assumption of a Gaussian distribution of water molecules diffusion, whereas this could not be exactly the case in vivo. It is, therefore, diffusion kurtosis imaging (DKI) to try to curtain this imperfection. There are studies which suggest the
          advantages of DKI in pathological classification, such as diagnosis of conventional glioma grading (HGG vs. LGG) by Tietze et al., <this.Cite name='2'/> and WHO gliomas grading (grade 2-4) and IDH-mutation status prediction by Bisdas et al. <this.Cite name='Nature'/> The work of this paper follows the latter study in the IDH-mutation status prediction respect. </p>
        <p>IDH stands for isocitrate dehydrogenase. Bisdas et al. introduce an approach of leveraging first-order statistics and texture feature extraction by MR8 <this.Cite name='3'/> filters to classify IDH wild-type and IDH-mutant tumours. In their study, they computed DKI first-order statistics biomarkers (mean, median, standard deviation, kurtosis, 5th and 95th percentile) from response images from MR8 filters, and then combined and put biomarkers into
          support vector machine (SVM) to perform training and to make prediction. Fluid Attenuated Inversion Recovery (FLAIR) images are MRI images cerebrospinal fluid (CSF) signal of which has been nullified. <this.Cite name='4'/> Both FLAIR and DKI images were processed, jointly and separately. Their results illustrate that experiments on DKI images solely achieved accuracy of 83.8% (sensitivity 0.96, specificity 0.55; for wild-type status as negative and
          mutant status as positive) and experiments on FLAIR images achieved accuracy of 73% (sensitivity 0.88, specificity 0.36), whilst, not surprisingly, the result was the same as DKI images solely when both modalities were taken as input. Examples of FLAIR and DKI images and their MR8 response images cited from <this.Cite name='Nature' inline={1}/> are shown in figure 1-2.</p>
        <p></p>

        <div className='fig-ct'>
          <Figure source='figures/FLAIR MR8.jpg' caption={
            <span>FLAIR image (A) and its MR8 responses (B-J).<this.Cite name='Nature'/></span>
          } width={500}/>
        </div>
        <div className='fig-ct'>
          <Figure source='figures/DKI MR8.jpg' caption={
            <span>DKI image (A) and its MR8 responses (B-J).<this.Cite name='Nature'/></span>
          } width={500}/>
        </div>

        <p>The above results not only suggest that it may be doable to predict IDH-mutation status by applying machine learning techniques to MRI images but also add evidence to proposing the potential of DKI. Yet, MR8 filters by which the features of MRI images were actuated were the texture detectors designed manually, instead of those being learned automatically by algorithm which modern machine learning methodology would prefer. Thus, experiments further to the previous study were to applied deep learning techniques to deeply learn image filters that can be used for the same purpose without subjective biases that could arise from the factitiousness of MR8 filters. </p>

        <p>To do so, convolutional neural networks (CNNs) are introduced. CNNs have credible capacity in image recognition and are widely used in computer vision tasks. The principle of CNN is to have multiple channels of filters (kernels) to extract feature from the image, and more importantly, those filters are learned by optimisation algorithms to fit the specific task automatically, which makes it apposite to this consideration. But CNN is a high variance algorithm that requires a large amount of data to train. Without an enough quantity of data, it is impossible for CNN to give decent result.</p>

        <p>In 2012, Menze et al. organised BraTS (brain tumour segmentation) challenge to gauge the status of automated brain tumour segmentation and they released a considerable amount of MRI data so as to draw attentions of researchers to take part in the challenges. <this.Cite name='5'/><this.Cite name='6'/> Since then, data for BraTS challenges were issued every year. BraTS2015 dataset is the data released for the challenge in the year 2015. It consists of 220 HGGs (high-grade gliomas) and 54 LGGs (low-grade gliomas). BraTS2015 data should fuel the deeply learning of filters in CNN. On this basis, experiments were designed and conducted to try to either support or extend the findings from the previous works.</p>

        <this.SectionTitle title='Literature review'/>
        <this.Subtitle title='Convolutional neural network'/>

        <p>Neural network is commonly considered a very powerful learning algorithm today. Its design was originally inspired by the mechanism of the human brain which, in a simile way of saying, is also a machine, yet an organic and electrochemical one, made up of neurons and their connectomes. Basically, the mathematical foundation of neural network algorithms can be represented by matrices arithmetic. Matrices of weights \Theta weigh the connections between
          layer <span className='math-inline'><MathJax math='l'/></span> and layer <span className='math-inline'><MathJax math='l+1'/></span>. Weighted values of each layer go through the activation function <span className='math-inline'><MathJax math='g(x)'/></span> before they partake in the next layer. The output of <span className='math-inline'><MathJax math='i^{th}'/></span> neuron in layer <span className='math-inline'><MathJax math='l+1'/></span> is
          <span className='math-block'>
            <MathJax math='a_i^{l+1}= g\left(\sum_{j=0}^{n-1} \Theta_{ij}^{(l)} a_j^{(l)}\right)\tag{1}'/>
          </span>
          For brevity, biases are not included in this equation.
        </p>
        <p>Within the category of neural network there is convolution neural network (CNN) which has been proven to be greatly efficient in computer vision because of its parameter sharing and connection sparsity properties. Convolution operation computes the dot product between the input and the filter which is the weights corresponding to <span className='math-inline'><MathJax math='\Theta'/></span> in equation (1), but for convolution
          the weight matrices <span className='math-inline'><MathJax math='\Theta'/></span> are shared across all the neurons from the previous layer. Convolutional layers in CNN function as feature extractor, while fully connected layers (represented as equation (1)) connect all the features globally and finally make inferences. Gradient descend is used in neural networks to optimise weights and biases for the training data.</p>

        <this.Subtitle title='Batch normalisation'/>
        <p>Batch normalisation <this.Cite name='7'/> is a terrific technique that have shown to be able to by many times speed up convergence of the model. This technique can neutralise the dissimilarity of distribution across layers, and thereby accelerates gradient. Batch normalisation can also improve prediction accuracy of image classification model, as specified in <this.Cite name='7' inline={1}/>. In short, it computes mean and variance of a mini batch and uses these two values to normalise the input of a layer batch-wise. When training, this statistic values were computed on a mini-batch; when testing, or in inference, it uses moving statistics to normalise the input. The mathematical base is as follow:
        <span className='math-block'>
          <MathJax math='\begin{split} & \mu=\frac{1}{m}\sum_{i=1}^mx_i \\ & \sigma=\frac{1}{m}\sum_{i=1}^m(x_i-\mu)^2 \\ & \widehat{x_i}=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} \\ & y_i=\gamma\widehat{x_i}+\beta \end{split}'/>
        </span>
        , where <span className='math-inline'><MathJax math='x_i'/></span> is input value to batch normalisation layer and <span className='math-inline'><MathJax math='y_i'/></span> is the output of this layer, both <span className='math-inline'><MathJax math='\gamma'/></span> and <span className='math-inline'><MathJax math='\beta'/></span> being trainable variables. The motive of batch normalisation is to shift the data-flow space to central to zero where the activation function is changing most drastically so that the difference between two values can be interpreted greater to the next layer.</p>

        <this.Subtitle title='Regularisation'/>
        <p>Overfitting is a severe problem in high variance algorithms like neural networks. With massive parameters, neural networks can almost always find a way to fit the training data decently (if the model is designed in a sensible way), but at the expense of generalisation to other data. To reduce overfitting, regularisation techniques need to be applied.</p>
        <p>Dropout regularisation <this.Cite name='8'/> is usually applied to fully connected layers in which the connection between neurons are very dense and the parameters are preponderant compared to convolutional layers which make fully connected layers the ringleader of overfitting. Dropout regularisation blocks a portion of neurons each time. This reduces the reliance on particular features. On the other hand, since only the weights and biases of unblocked neurons can be trained each time, training will take longer when dropout is applied. But also, since many neurons are blocked, computation only happens in those still active neurons. This makes training of each data flow faster.</p>
        <p>Moreover, L2 regularisation can penalise large weight values. This helps reduce the reliance of large weights and thus improve generalisation. L2 regularisation can be introduced in cross entropy loss function:
        <span className='math-block'>
          <MathJax math='J\left(\mathrm{\Theta}\right)=-\left[\frac{1}{m}\sum_{i=1}^{m}\left(y\prime^{\left(i\right)}\log{y^{\left(i\right)}}+\left(1-y\prime^{\left(i\right)}\right)\log {\left(1-y^{\left(i\right)}\right)}\right)\right]+\frac{\lambda}{2m}\sum_{j=1}^{n}\mathrm{\Theta}_j^2'/>
        </span>
        , where y^\prime is the ground truth label value, y is the predicted value and \lambda is the regularisation parameter.</p>

        <this.Subtitle title='Support vector machine'/>
        <p>Though many powerful modern machine learning algorithms have been developed today, support vector machine (SVM) has stood the test of time. SVM chooses landmarks from the training data and measures the similarity between a sample and the landmarks. By manually controlling the hyper-parameters, users can determine how SVM is to maximise its decision boundary between two classes.</p>
        <p>Therefore, SVM is a highly controllable binary classifier. Its objective function can be described as: <this.Cite name='9'/>
          <span className='math-block'>
            <MathJax math='J(\theta)\ =\ C\sum_{i=1}^{m}{\left[y^{(i)}{cost}_1\left(\theta^Tx^{(i)}\right)+\left(1-y^{(i)}\right){cost}_0\left(\theta^Tx^{(i)}\right)\right]+\frac{1}{2}\sum_{i=1}^{n}\theta_j^2}'/>
          </span>
          , where <span className='math-inline'><MathJax math='C'/></span> is the regularisation parameter that trades off the width of decision boundary and the extent of misclassification. Another hyper-parameter (<span className='math-inline'><MathJax math='\gamma'/></span>) comes from the RBF (radial basis function) kernel also known as Gaussian kernel:
          <span className='math-block'>
            <MathJax math="k\left(x,x\prime\right)=exp\left(-γ\|x-x\prime\|^2\right)"/>
          </span>
          <span className='math-inline'><MathJax math='\gamma'/></span> is basically the reciprocal of variance in Gaussian function. It determines the measurement of similarity of feature <span className='math-inline'><MathJax math='x'/></span> to landmark <span className='math-inline'><MathJax math='x\prime'/></span>.
        </p>

        <this.Subtitle title='Nested leave-one-out cross validation'/>
        <p>To deal with insufficiency of data samples, and to select hyper-parameters in SVM, nested leave one out cross validation (nested LOOCV) plays a role. Nested LOOCV was used to select hyperparameters C and \gamma in SVM. Nested LOOCV has an outer loop to perform LOOCV and an inner loop which does grid search over ranges of hyperparameters and then adopts different combination of parameters to perform LOOCV. The combination of parameters that yields best result in each inner loop will be fed to the outer loop.</p>
        <p>Nested LOOCV is usually used together with SVM model. The pseudocode for selecting hyper-parameters C and \gamma for SVM is shown below: <this.Cite name='10'/></p>
        <div style={{}}>
          <img src="figures/loocv.jpg" alt="LOOCV" style={{width:350, maxWidth: '90%'}}/>
        </div>
        <p>LOOCV is the one of the most unbiased way of validating models. However, because of its high complexity, it is more useful in cases where the dataset is reasonably small. Nested LOOCV is the variant of LOOCV which can perform hyper-parameter choosing.</p>

        <this.SectionTitle title='Experiments'/>
        <this.Subtitle title='2-D convolution'/>
        <p>Some experiments on 2-dimensional level CNN had already been done beforehand. In those experiments, three networks were trained:</p>
        <p>Network 1 is the pretrained VGG16 <this.Cite name='11'/> model trained to recognise objects in images and trained by the creators of the network architecture.</p>
        <p>Network 2 is the model trained from scratch on MR images to recognise tumour texture from non-tumour texture. Weights were randomly initialised by the training algorithm at the start of training.</p>
        <p>Network 3 is the tweaked model trained to recognise tumour texture from non-tumour texture. The network weights were initialised with the weights from the pretrained VGG16 model and was then further trained with MR images.</p>
        <p>The second and third network were trained on BraTS2015 dataset to classify tumour patches from non-tumour patches, (a patch is a small crop of a slice to the 3-D MRI image). Network 1 was pretrained, so stayed the same. Network 2 reported an accuracy of 81.14% while network 3 reached a better accuracy of 93.07%.</p>
        <p>MRI (DKI and FLAIR) images the IDH-mutation statuses of which were to be predicted were then passed through the hidden layers of these networks. From the response images of each layer first-order statistics values were produced for support vector machine classifier, similar to the approach as in the previous study.</p>
        <p>Though network 2 would be inferior while performing tumour and non-tumour classification task, when it came to IDH¬-mutation status prediction, it seemed to outperform network 3 by reaching the high accuracy (89.19%) more frequently. Comparison of the two modalities (DKI and FLAIR) in this experiment had also agreed with the one in the previous study, except for the fact that the outcome from FLAIR images was not that unacceptable as with still a very high accuracy of 86.49%, while, for DKI, the number was 91.89%. 91.89% accuracy is a very astonishing result. This means in one experiment, only 3 in 37 samples were misclassified by the algorithm. Sensitivity for this result was 0.96 and specificity was 0.82.</p>

        <this.Subtitle title='3-D convolution'/>
        <p>Extension to 3-dimensional level is the main subject of this paper. Therefore, this is to be discussed in detail.</p>
        <p>Given the observed results, it was known that filters from CNN provided better prediction accuracy for the data at hand. But the methods of that study were nonetheless limited to 2-dimension, whereas the MRI data were collected originally in 3-dimensional form (e.g. 276×320×43 for UCL FLAIR images, 128×128×25 for UCL DKI images, 240×240×155 for BraTS2015 FLAIR images) – an image is plotted in voxels in figure 3 to illuminate a 3-dimensional view of the data. It is more natural to treat the MRI data as 3-D than 2-D images. 3-D convolution is no different from 2-D convolution in principle. If using all 3 by 3 by 3 kernels, 3-D convolution has 3 times more parameters than 2-D convolution, and the third dimension of the image itself also brings in cubic memory requirement. The increased volume would cause memory shortage if keeping the architecture (more precisely depth and number of channels) same as 2-D convolution.</p>
        <div className='fig-ct'>
          <Figure source='figures/voxels.jpg' caption={
            <span>Image voxels example. An image of size 155×240×249 contains nearly 9 million voxels. Only a relatively small fraction of those in this example were plotted. Voxel values are visualised as opacity of the point on the graph. The tumour space here can be easily discerned. That is the dense region on the lower-right. But also see that opaque point does not necessarily belong to tumour region.</span>
          } width={400}/>
        </div>

        <this.Subtitle title='CNN models' level={3}/>
        <p>Two networks were designed for the experiments. One has a VGG-like architecture with serial 3×3 convolutional layers and 2×2 max pooling layers (see figure 4). Though many deep convolutional models have been developed since the advent of VGG net <this.Cite name='11'/>, VGG is still widely used. Its design uses more layers of 3×3 convolution kernels to replace 5×5 or 7×7 kernels to keep a large receptive field but at the same time reduce computation. There are lots of machine learning models developed on top of VGG or its variants. The design herein follows the VGG practice where max pooling layers downsize the input image to 7×7 feature maps which are then flattened and injected to fully-connected layers. To reduce computation and memory consumption, batch normalisation layers are added in a gap of two layers instead of in every layer. This turned out not affecting the performance in terms of prediction accuracy at all. There are 12 trainable layers in this model. Six are convolutional layers. This network could produce 6 layers of response images. The total number of parameters was counted as 4,833,250.</p>
        <div className='fig-ct'>
          <Figure source='figures/network1.jpg' caption={
            <span>Network one. Three-dimensional neural network using 3×3×3 filters. Seen here from left to right are (1) input layer, (2) convolution
              layer, (3) batch normalisation layer, (4) & (5) convolution layer, (6) batch normalisation layer, (7) max pooling layer, (8) convolution layer, (9) batch normalisation layer, (10) & (11) convolution layer, (12) batch normalisation layer, (13) max pooling layer, (14) & (15) fully connected layer, (16) dropout layer, (17) softmax layer, (18) output layer.</span>
          } width={800}/>
        </div>
        <p>The other model (network two) was implemented in light of the concept of deep inception network by Szegedy et al. <this.Cite name='12'/> The incentive of inception net is to reduce the number of parameters but to also deepen the network. In the original design, average pooling is used instead of fully-connected layers. This indeed reduces a lot of computational operations, but it also causes the decomposition of spatial information. For tumour detection task, it is important to have the capacity to locate the tumour boundary in a patch. So, in this case, its doubling of channels in each inception unit and introducing of fully-connected layers in effect resulted in an increase of parameters, as the number counted as 18,555,810. Likewise, the output of the last convolutional layers is 7×7 feature maps and it is the input of two fully-connected layers. Unlike VGG net, inception net is prone to ever mix up the spatial information of the objects in an image even though without average pooling layers in the final, but it has been proven to be more competent for classification tasks with its deep and elaborate design and reduction of parameters, which is not in use here. The configuration of this network is simplified in figure 5. In figure 5, there are two inception modules (named inception unit in the figure). Convolutional pooling functions as the pooling layer similar to max pooling in VGG net, which is for dimensionality reduction. This is introduced in Inception-V4 in <this.Cite name='13'/>.</p>
        <div className='fig-ct'>
          <Figure source='figures/network2.jpg' caption={
            <span>Network two. Three-dimensional neural network applying inception architecture. Seen here from left to right are (1) input layer, (2) convolution layer, (3) batch normalisation layer, (4) convolution layer, (5) batch normalisation layer, (6) convolutional pool module, (7) inception module, (8) batch normalisation layer, (9) convolutional pool module, (10) inception module, (11) batch normalisation layer, (12) & (13) fully connected layer, (14) dropout layer, (15) softmax layer, (16) output layer, where configuration of convolutional pool module is on the left-bottom and configuration of inception module is on the right-bottom.</span>
          } width={800}/>
        </div>
        <p>Because the small size of input image, deeper neural networks do not help to extend receptive field of a neuron, but will produce more abstract features and become more difficult to train. For 3×3 convolution kernels and halving reduction layers (e.g. 2×2 with stride 2 max pooling layers), to compute receptive field of CNN, the following equation can be used:
          <span className='math-block'>
            <MathJax math="r=\sum_{n=1}^{k}{2^{n-1}l_n}"/>
          </span>
          , where <span className='math-inline'><MathJax math='k'/></span> is the number of reduction layers and <span className='math-inline'><MathJax math='l'/></span> is the number of convolutional layers between <span className='math-inline'><MathJax math='n^{th}'/></span> and <span className='math-inline'><MathJax math='{(n+1)}^{th}'/></span> reduction layers,
           <span className='math-inline'><MathJax math='r'/></span> being the radius of the reduction field.
        </p>
        <p>If reception field is larger, the neurons detecting tumour region will be affected by more voxels outside the tumour region. This causes more undesirable boundary effect and affects the texture of the response tumour region. Therefore, the depths of the networks are designed conservatively in the experiments herein. On the other hand, in this case, the convergence was found faster when the implementation of batch normalisation was after the activation function, unlike the practice in the original paper or the practice in ResNet <this.Cite name='14'/> where this technique is massively used before activations.</p>

        <this.Subtitle title='UCL dataset' level={3}/>
        <p>UCL MR images is a set of 37 multi-parametric MR images collected by the clinicians at the University College London of 37 anonymised patients, mean age 63.2 years ± 7.6 [standard deviation], age range 27–76 years). The data included the tumour grade, IDH-mutation status and the tumour segmentation. Of the concern of this paper is IDH-mutation status. 26 of the patients have mutated IDH and 11 patients have the wild-type IDH.</p>

        <this.Subtitle title='BraTS2015 dataset' level={3}/>
        <p>BraTS2015 consists of 220 HGGs (high-grade gliomas) and 54 LGGs (low-grade gliomas) brain MRI samples. A sample from BraTS2015 data consists of images of 4 different modalities, where different modalities highlight different tumour structures (i.e. edema, necrosis, enhancing tumour and non-enhancing tumour), and one segmentation label which is delineated by human experts, indicating the tumour area of four different structures against other organic matter in the skull.</p>

        <this.Subtitle title='BraTS data pre-processing' level={3}/>
        <p>As alluded to above, to obtain learned filters from CNN, large quantity of data is required. BraTS2015 dataset provides brain MRI images which can be used in training the networks, so that ideally the parameters which constitute filters are trained to pick up features of the tumour texture to distinguish them from non-tumour texture.</p>
        <p>Data from BraTS2015 were split into training, validation and test set. A programme to sample image patches took the three-dimension images as input and produced the equal number of patches for tumour and non-tumour tissue, having size of 28×28×14. Using ‘max-min’ normalisation:
          <span className='math-block'>
            <MathJax math="z_i=\frac{x_i-min(x)}{max(x)-min(x)}"/>
          </span>
          , cubic patches were normalised and prepared for training. Overall, there were 6564 patches produced.
        </p>

        <this.Subtitle title='Implementation' level={3}/>
        <p>The two different configurations of CNN were both implemented using TensorFlow (python). Each training experiment roughly took 2 hours of running time for a single GPU (Tesla K80). Adam optimisation was used in mini batch gradient descend. Batch size was 128. Learning rate was set to 0.0001 at the beginning and after 200 iteration changed to 0.00001. Dropout rate was set to 0.5. L2 regularisation parameter \lambda was set to 0.1. These parameters are the same for two networks. Training’s were stopped when training loss and validation loss no longer decreased. Generally, an experiment of training went through 10 epochs.</p>

        <this.Subtitle title='Training and testing' level={3}/>
        <p>Data were separated into training set (60%), validation set (20%) and testing set (20%). During training, training loss and accuracy as well as validation loss and accuracy were observed and used to determine where to stop. An example of experiment observation is shown in figure 6-9.</p>

        <div className='fig-ct'>
          <Figure source='figures/accuracy.png' caption={
            <span>Accuracy lines from network 1. Blue: training accuracy; green: validation accuracy.</span>
          } width={360}/>
          <Figure source='figures/loss.png' caption={
            <span> Loss lines from network 1. Blue: training loss; green validation loss.</span>
          } width={360}/>
        </div>
        <div className='fig-ct'>
          <Figure source='figures/accuracy2.png' caption={
            <span>Accuracy lines from network 2. Blue: training accuracy; green: validation accuracy.</span>
          } width={360}/>
          <Figure source='figures/loss2.png' caption={
            <span>Loss lines from network 2. Blue: training loss; green validation loss.</span>
          } width={360}/>
        </div>

        <p>Figure 6 and 7 show that the first model is easier to train since, for instance, its loss line quickly hits the ground in roughly 300 iterations. But its test lines are not in accordance with the train lines. This may suggest that this outcome is the effect of overfitting. On the contrary, figures 8-9 from network 2 show a more stable pattern, and the training line and validation line travel close to each other. By comparing the four charts, it is not venturesome to conclude that the inception net has better properties for this task, because of its stability and convergence. In fact, the inception configuration achieved an accuracy of 95.78% for test set, by contrast with the accuracy of 93.18% for the VGG-like configuration. But basically, these two networks both achieved high performance.</p>

        <this.Subtitle title='Feature extraction' level={3}/>
        <p>To get response images from convolutional filters, DKI and FLAIR images from UCL dataset were input into the two models. After each convolutional layer, the response tensor was the feature maps of the input image, and thereby convolutional features were extracted. Response tensors from a particular layer were concatenated with tensors from all previous layers or would become the feature maps of this layer without concatenation. Figure 10 illustrates a slice of 3-D FLAIR image and its 64-channel responses from the last second layer in network 1. Figure 11 is the responses for DKI images. Since the image is three-dimensional, only a slice of it is taken for demonstration purpose.</p>

        <div className='fig-ct'>
          <Figure source='figures/FLAIR CONV.jpg' caption={
            <span>An example of FLAIR image and its 64-channel responses. Shown are one slice of the 3-dimensional image. A is the original image slice. B is the 11th channel response within the 64 channels. The red curves are the tumour.</span>
          } width={450}/>
        </div>
        <div className='fig-ct'>
          <Figure source='figures/DKI CONV.jpg' caption={
            <span>An example of DKI image and its 64-channel responses. Shown are one slice of the 3-dimensional image. A is the original image slice. B is the 11th channel response within the 64 channels. The red curves are the tumour.</span>
          } width={450}/>
        </div>
        <p>For network 1 (VGG-like), there were 6 convolutional layers, so 6 layers of response images were obtained. For network 2, both ‘inception unit’ layers and ‘convolutional pooling’ layers were account for convolutional filters; there were still 6 layers. Response images of different layers produced first-order statistic values (mean, median, standard deviation, kurtosis, 5th and 95th percentile) by each layer alone or by concatenating with the prior layers. Lengths of feature vector increased quickly in the concatenation case. For the instance of network 2, the vector length of the last convolutional layer was 4512. Using equation (2) again, feature vectors were normalised dimension-wise.</p>

        <this.Subtitle title='Support vector machine classification' level={3}/>
        <p>Feature vectors were then input to support vector machine. LIBSVM <this.Cite name='15'/> with RBF kernel was used, following the practice in <this.Cite name='Nature' inline={1}/>. LIBSVM is a popular opensource library for SVM and is stable and easy to use. In need of selecting SVM hyper-parameters as well as dealing with the limited number of samples, a nested leave-one-out cross validation (LOOCV) setting was used with the inner loop for parameter selection and the outer for model assessment. This ensure an unbiased estimation of the true error, though the amount of data is deficient. In addition, since training of SVM is very sensitive to class imbalance, positive class and negative class were weighted by
          <span className='math-block'>
            <MathJax math="C_p={\left(N^++N^-\right)}/{2N^+}\times\ C"/>
          </span>
        , and
          <span className='math-block'>
            <MathJax math="C_n={\left(N^++N^-\right)}/{2N^-\times C}"/>
          </span>
        respectively, where <span className='math-inline'><MathJax math='N^+'/></span> is the number of positive samples and <span className='math-inline'><MathJax math='N^-'/></span> is the number of negative samples.
        </p>
        <p>The optimal parameters (<span className='math-inline'><MathJax math='C, \gamma'/></span>) for SVM were found by grid search in inner loops. Each optimal parameter set corresponds to one inner loop and is applied to testing the left one in the outer loop. </p>

        <this.SectionTitle title='Results'/>
        <p>Experiments for IDH-mutation status prediction were conducted on UCL dataset of 37 samples. Similar to the previous works, feature extraction on MRI images assisted by SVM yield high prediction accuracy on IDH-mutation status prediction. High accuracies were appeared when the features were extracted from the last two layers of the networks. The main difference is, in those experiments here, FLAIR and DKI with two different networks resulted in a same accuracy of 86.49% (sensitivity 0.96, specificity 0.64). It seems that the high accuracy (91.89%) in 2-D convolution and the difference between modalities are not the case for this 3-D extension. But this result seems to have advantages over that from MR8 filters.</p>

        <p>Table 1 lists the best results from the experiments, including MR8, 2-D and 3-D filters settings.</p>
        <MyTable width={380} caption={
          <span>Comparing results from three sets of experiments. Row two lists the modalities used when the accuracy was achieved.</span>
          }
          thead={
            <thead>
              <tr>
                <th></th>
                <th>MR8</th>
                <th>2-D CONV</th>
                <th>3-D CONV</th>
              </tr>
            </thead>
          } tbody={
            <tbody>
            <tr>
              <th>ACHIEVED ACCURACY</th>
              <td>83.78%</td>
              <td>91.89%</td>
              <td>86.49%</td>
            </tr>
            <tr>
              <th>MODALITY</th>
              <td>DKI</td>
              <td>DKI</td>
              <td>DKI and FLAIR</td>
            </tr>
          </tbody>
        }/>
        <p>DKI performed better than FLAIR in MR8 and 2-D convolution cases, but here they played a draw. Also, the best achieved accuracy here is in between of that of the other two experiments.</p>

        <this.SectionTitle title='Discussion'/>
        <p>Experiments began with training neural networks to classify tumour and non-tumour textures. Performance of the deep learning algorithm for this tumour detection task was significantly improved when the set up was extended from 2-D to 3-D to conform the 3-dimension nature of the MRI images. The advantages of inception net compared to VGG net here are also salient. This may suggest that 3-D inception architecture would be a better choice in general than VGG for MRI images. One reason for that could be the diversity of architecture captures more detailed texture features. This could be an important finding in this study especially when where those images function is in medical area, where the concerns of stability and accuracy cannot be more serious. Future experiments may take place on this point. But 2-D VGG did outperform 3-D inception net in IDH-mutation status prediction task. Also, in tumour segmentation experiments, which is beyond the scope of this paper, VGG manifested itself to be a better choice over Inception. It is difficult to say one is better than another by comparing just a few experiments and with handful of data. But it is also precisely this miscellaneous nature of these technologies that makes various innovative sometimes audacious ideas like automation of medical diagnosis possible.</p>
        <p>From another perspective, brain MRI images were not designed for classifying this subtle pathological nuance. Its resolution is not quite enough for reflecting many latent details lying inside the tissue. More attention should be drawn to developing technologies for higher resolution medical images. Otherwise, in spite of rapid development of machine learning and image processing methods, if there is no match-up quality of images, the achievement would still be limited.</p>
        <p>By means of machine learning, algorithms for IDH-mutation status prediction can be developed. Also, the result seems promising. Be that as is may, it is hardly convincing for a conclusion to be drawn from this quite small dataset, though the outcome is good, and the methods are substantiated. Because of this, the difference between these three studies are not in conflict with each other, but quite the contrary, its evidencing and questioning aspects necessitate the collection of more data in the future, notwithstanding the author’s best aspiration that the disease will no longer plague us and there will be no more data to collect.</p>


        <this.SectionTitle title='References'/>
        <References references={this.references}/>

        <div className='windows-frame-size-keeper' ref='windowsSizeKeeper'></div>
        <Windows ref='windows' parent={this}/>
      </div>
    )
  }
}
export default Content
