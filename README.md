# Top 10 Computer Vision Papers 2020
## The top 10 computer vision papers in 2020 with video demos, articles, code, and paper reference.

Even with everything that happened in the world this year, we still had the chance to see a lot of amazing research come out. Especially in the field of artificial intelligence and more precisely computer vision. More, many important aspects were highlighted this year, like the ethical aspects, important biases, and much more. Artificial intelligence and our understanding of the human brain and its link to AI is constantly evolving, showing promising applications in the soon future, which I will definitely cover.

Here are my top 10 of the most interesting research papers of the year in computer vision, in case you missed any of them. In short, it is basically a curated list of the latest breakthroughs in AI and CV with a clear video explanation, link to a more in-depth article, and code (if applicable). Enjoy the read, and let me know if I missed any important papers in the comments, or by contacting me directly on LinkedIn!

**The complete reference to each paper is listed at the end of this repository.**

Maintainer - [louisfb01](https://github.com/louisfb01)

*Feel free to message me any great papers I missed to add to this repository on bouchard.lf@gmail.com*

***Tag me on Twitter [@Whats_AI](https://twitter.com/Whats_AI) or LinkedIn [@Louis (What's AI) Bouchard](https://www.linkedin.com/in/whats-ai/)  if you share the list!***

### Watch a complete computer vision 2020 rewind in 5 minutes

[![Watch the video](https://imgur.com/5ifuFAd.png)](https://youtu.be/CP3E9Iaunm4)

---

### If you are interested in AI research, here is another great repository for you:
A curated list of the latest breakthroughs in AI by release date with a clear video explanation, link to a more in-depth article, and code.

[2020: A Year Full of Amazing AI papers- A Review](https://github.com/louisfb01/Best_AI_paper_2020)


## The Full List
- [Sea-thru: A Method For Removing Water From Underwater Images [1]](#1)
- [Neural circuit policies enabling auditable autonomy [2]](#2)
- [NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis [3]](#3)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection [4]](#4)
- [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models [5]](#5)
- [Image GPT - Generative Pretraining from Pixels [6]](#6)
- [DeepFaceDrawing: Deep Generation of Face Images from Sketches [7]](#7)
- [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization [8]](#8)
- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow [9]](#9)
- [Learning Joint Spatial-Temporal Transformations for Video Inpainting [10]](#10)
- [Old Photo Restoration via Deep Latent Space Translation [Bonus 1]](#Bonus1)
- [Is a Green Screen Really Necessary for Real-Time Portrait Matting? [Bonus 2]](#Bonus2)
- [DeOldify [Bonus 3]](#Bonus3)
- [Paper references](#references)

---

## Sea-thru: A Method For Removing Water From Underwater Images [1]<a name="1"></a>
Have you ever wondered how the ocean would look like without water Remove this blue-green tint of the underwater pictures, and still have the true colors of a coral reef? Well, using computer vision and machine learning algorithms, researchers from the University of Haifa were able to accomplish exactly that!

* Short Video Explanation:

[![Watch the video](https://imgur.com/h9EFruo.png)](https://youtu.be/CP3E9Iaunm4)
* [This AI Removes the Water from Underwater Images!](https://medium.com/towards-artificial-intelligence/this-ai-removes-the-water-from-underwater-images-d277281bcd0f) - Short Read
* [Sea-thru: A Method For Removing Water From Underwater Images](https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf) - The Paper
* [Click here for the Sea Thru code](https://github.com/jgibson2/sea-thru) - The Code


## Neural circuit policies enabling auditable autonomy [2]<a name="2"></a>
Researchers from IST Austria and MIT have successfully trained a self-driving car using a new artificial intelligence system based on the brains of tiny animals, such as threadworms. They achieved that with only a few neurons able to control the self-driving car, compared to the millions of neurons needed by the popular deep neural networks such as Inceptions, Resnets, or VGG. Their network was able to completely control a car using only 75 000 parameters, composed of 19 control neurons, rather than millions!

* Short Video Explanation:

[![Watch the video](https://imgur.com/86EtfbT.png)](https://youtu.be/wAa358pNDkQ)
* [A New Brain-inspired Intelligent System Drives a Car Using Only 19 Control Neurons!](https://medium.com/towards-artificial-intelligence/a-new-brain-inspired-intelligent-system-drives-a-car-using-only-19-control-neurons-1ed127107db9) - Short Read
* [Neural circuit policies enabling auditable autonomy](https://www.nature.com/articles/s42256-020-00237-3.epdf?sharing_token=xHsXBg2SoR9l8XdbXeGSqtRgN0jAjWel9jnR3ZoTv0PbS_e49wmlSXvnXIRQ7wyir5MOFK7XBfQ8sxCtVjc7zD1lWeQB5kHoRr4BAmDEU0_1-UN5qHD5nXYVQyq5BrRV_tFa3_FZjs4LBHt-yebsG4eQcOnNsG4BenK3CmBRFLk%3D) - The Paper
* [Click here for the NCP code](https://github.com/mlech26l/keras-ncp) - The Code


## NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis [3]<a name="3"></a>
This new method is able to generate a complete 3-dimensional scene and has the ability to decide the lighting of the scene. All this with very limited computation costs and amazing results compared to previous approaches.

* Short Video Explanation:

[![Watch the video](https://imgur.com/w3QnN6g.png)](https://youtu.be/ZkaTyBvS2w4)
* [Generate a Complete 3D Scene Under Arbitrary Lighting Conditions from a Set of Input Images](https://medium.com/what-is-artificial-intelligence/generate-a-complete-3d-scene-under-arbitrary-lighting-conditions-from-a-set-of-input-images-9d2fbce63243) - Short Read
* [NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis](https://arxiv.org/abs/2012.03927) - The Paper
* [Click here for the NeRV code *(coming soon)*](https://people.eecs.berkeley.edu/~pratul/nerv/) - The Code


## YOLOv4: Optimal Speed and Accuracy of Object Detection [4]<a name="4"></a>
This 4th version has been recently introduced in April 2020 by Alexey Bochkovsky et al. in the paper "YOLOv4: Optimal Speed and Accuracy of Object Detection". The main goal of this algorithm was to make a super-fast object detector with high quality in terms of accuracy.

* Short Video Explanation:

[![Watch the video](https://imgur.com/ShqhwQl.png)](https://youtu.be/CtjZFkO5RPw)
* [The YOLOv4 algorithm | Introduction to You Only Look Once, Version 4 | Real-Time Object Detection](https://medium.com/what-is-artificial-intelligence/the-yolov4-algorithm-introduction-to-you-only-look-once-version-4-real-time-object-detection-5fd8a608b0fa) - Short Read
* [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) - The Paper
* [Click here for the Yolo v4 code](https://github.com/AlexeyAB/darknet) - The Code


## PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models [5]<a name="5"></a>
This new algorithm transforms a blurry image into a high-resolution image!
It can take a super low-resolution 16x16 image and turn it into a 1080p high definition human face! You don't believe me? Then you can do just like me and try it on yourself in less than a minute! But first, let's see how they did that.

* Short Video Explanation:

[![Watch the video](https://imgur.com/2R9Yhgk.png)](https://youtu.be/cgakyOI9r8M)
* [This AI makes blurry faces look 60 times sharper](https://medium.com/what-is-artificial-intelligence/this-ai-makes-blurry-faces-look-60-times-sharper-7fcd3b820910) - Short Read
* [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models](https://arxiv.org/abs/2003.03808) - The Paper
* [Click here for the PULSE code](https://github.com/adamian98/pulse) - The Code


## Image GPT - Generative Pretraining from Pixels [6]<a name="6"></a>
A good AI, like the one used in Gmail, can generate coherent text and finish your phrase. This one uses the same principles in order to complete an image! All done in an unsupervised training with no labels required at all!

* Short Video Explanation:

[![Watch the video](https://imgur.com/4RKhkL0.png)](https://youtu.be/FwXQ568_io0)
* [This AI Can Generate the Other Half of a Picture Using a GPT Model](https://medium.com/towards-artificial-intelligence/this-ai-can-generate-the-pixels-of-half-of-a-picture-from-nothing-using-a-nlp-model-7d7ba14b5522) - Short Read
* [Image GPT - Generative Pretraining from Pixels](https://openai.com/blog/image-gpt/) - The Paper
* [Click here for the OpenAI's Image GPT code](https://github.com/openai/image-gpt) - The Code


## DeepFaceDrawing: Deep Generation of Face Images from Sketches [7]<a name="7"></a>
You can now generate high-quality face images from rough or even incomplete sketches with zero drawing skills using this new image-to-image translation technique! If your drawing skills as bad as mine you can even adjust how much the eyes, mouth, and nose will affect the final image! Let's see if it really works and how they did it.

* Short Video Explanation:

[![Watch the video](https://imgur.com/NJiJ3ny.png)](https://youtu.be/djXdgCVB0oM)
* [AI Generates Real Faces From Sketches!](https://medium.com/what-is-artificial-intelligence/ai-generates-real-faces-from-sketches-8ccbac5d2b2e) - Short Read
* [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf) - The Paper
* [Click here for the DeepFaceDrawing code](https://github.com/IGLICT/DeepFaceDrawing-Jittor) - The Code


## PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization [8]<a name="8"></a>
This AI Generates 3D high-resolution reconstructions of people from 2D images! It only needs a single image of you to generate a 3D avatar that looks just like you, even from the back!

* Short Video Explanation:

[![Watch the video](https://imgur.com/btyo32w.png)](https://youtu.be/ajWtdm05-6g)
* [AI Generates 3D high-resolution reconstructions of people from 2D images | Introduction to PIFuHD](https://medium.com/towards-artificial-intelligence/ai-generates-3d-high-resolution-reconstructions-of-people-from-2d-images-introduction-to-pifuhd-d4aa515a482a) - Short Read
* [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](https://arxiv.org/pdf/2004.00452.pdf) - The Paper
* [Click here for the PiFuHD code](https://github.com/facebookresearch/pifuhd) - The Code


## RAFT: Recurrent All-Pairs Field Transforms for Optical Flow [9]<a name="9"></a>
ECCV 2020 Best Paper Award Goes to Princeton Team. They developed a new end-to-end trainable model for optical flow. Their method beats state-of-the-art architectures' accuracy across multiple datasets and is way more efficient. They even made the code available for everyone on their Github!

* Short Video Explanation:

[![Watch the video](https://imgur.com/VdcyRAE.png)](https://youtu.be/OSEuYBwOSGI)
* [ECCV 2020 Best Paper Award | A New Architecture For Optical Flow](https://medium.com/towards-artificial-intelligence/eccv-2020-best-paper-award-a-new-architecture-for-optical-flow-3298c8a40dc7) - Short Read
* [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf) - The Paper
* [Click here for the RAFT code](https://github.com/princeton-vl/RAFT) - The Code


## Learning Joint Spatial-Temporal Transformations for Video Inpainting [10]<a name="10"></a>
This AI can fill the missing pixels behind a removed moving object and reconstruct the whole video with way more accuracy and less blurriness than current state-of-the-art approaches!

* Short Video Explanation:

[![Watch the video](https://imgur.com/pKAD8Zu.png)](https://youtu.be/MAxMYGoN5U0)
* [This AI takes a video and fills the missing pixels behind an object!](https://medium.com/towards-artificial-intelligence/this-ai-takes-a-video-and-fills-the-missing-pixels-behind-an-object-video-inpainting-9be38e141f46) - Short Read
* [Learning Joint Spatial-Temporal Transformations for Video Inpainting](https://arxiv.org/abs/2007.10247) - The Paper
* [Click here for this Video Inpainting code](https://github.com/researchmm/STTN?utm_source=catalyzex.com) - The Code


## Old Photo Restoration via Deep Latent Space Translation [Bonus 1]<a name="Bonus1"></a>
Imagine having the old, folded, and even torn pictures of your grandmother when she was 18 years old in high definition with zero artifacts. This is called old photo restoration and this paper just opened a whole new avenue to address this problem using a deep learning approach.

* Short Video Explanation:

[![Watch the video](https://imgur.com/cnds8hi.png)](https://youtu.be/QUmrIpl0afQ)
* [Old Photo Restoration using Deep Learning](https://medium.com/towards-artificial-intelligence/old-photo-restoration-using-deep-learning-47d4ab1bdc4d) - Short Read
* [Old Photo Restoration via Deep Latent Space Translation](https://arxiv.org/pdf/2009.07047.pdf) - The Paper
* [Click here for the Old Photo Restoration code](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life?utm_source=catalyzex.com) - The Code


## Is a Green Screen Really Necessary for Real-Time Portrait Matting? [Bonus 2]<a name="Bonus2"></a>
Human matting is an extremely interesting task where the goal is to find any human in a picture and remove the background from it. It is really hard to achieve due to the complexity of the task, having to find the person or people with the perfect contour. In this post, I review the best techniques used over the years and a novel approach published on November 29th, 2020. Many techniques are using basic computer vision algorithms to achieve this task, such as the GrabCut algorithm, which is extremely fast, but not very precise.

* Short Video Explanation:

[![Watch the video](https://imgur.com/EXMOzI9.png)](https://youtu.be/rUo0wuVyefU)
* [High-Quality Background Removal Without Green Screens](https://medium.com/datadriveninvestor/high-quality-background-removal-without-green-screens-8e61c69de63) - Short Read
* [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/pdf/2011.11961.pdf) - The Paper
* [Click here for the MODNet code](https://github.com/ZHKKKe/MODNet) - The Code


## DeOldify [Bonus 3]<a name="Bonus3"></a>
DeOldify is a technique to colorize and restore old black and white images or even film footage. It was developed and is still getting updated by only one person Jason Antic. It is now the state of the art way to colorize black and white images, and everything is open-sourced, but we will get back to this in a bit.

* Short Video Explanation:

[![Watch the video](https://imgur.com/jGOxFl1.png)](https://youtu.be/1EP_Lq04h4M)
* [This AI can Colorize your Black & White Photos with Full Photorealistic Renders! (DeOldify)](https://medium.com/towards-artificial-intelligence/this-ai-can-colorize-your-black-white-photos-with-full-photorealistic-renders-deoldify-bf1eed5cb02a) - Short Read
* [Click here for the DeOldify code](https://github.com/jantic/DeOldify) - The Code


---

***Tag me on Twitter [@Whats_AI](https://twitter.com/Whats_AI) or LinkedIn [@Louis (What's AI) Bouchard](https://www.linkedin.com/in/whats-ai/)  if you share the list!***

---

## Paper references<a name="references"></a>

[1] Akkaynak, Derya & Treibitz, Tali. (2019). Sea-Thru: A Method for Removing Water From Underwater Images. 1682–1691. 10.1109/CVPR.2019.00178.

[2] Lechner, M., Hasani, R., Amini, A. et al. Neural circuit policies enabling auditable autonomy. Nat Mach Intell 2, 642–652 (2020). https://doi.org/10.1038/s42256-020-00237-3

[3] P. P. Srinivasan, B. Deng, X. Zhang, M. Tancik, B. Mildenhall, and J. T. Barron, "Nerv: Neural reflectance and visibility fields for relighting and view synthesis," in arXiv, 2020.

[4] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, Yolov4: Optimal speed and accuracy of object detection, 2020. arXiv:2004.10934 [cs.CV].

[5] S. Menon, A. Damian, S. Hu, N. Ravi, and C. Rudin, Pulse: Self-supervised photo upsampling via latent space exploration of generative models, 2020. arXiv:2003.03808 [cs.CV].

[6] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, "Generative pretraining from pixels," in Proceedings of the 37th International Conference on Machine Learning, H. D. III and A. Singh, Eds., ser. Proceedings of Machine Learning Research, vol. 119, Virtual: PMLR, 13–18 Jul 2020, pp. 1691–1703. [Online].

[7] S.-Y. Chen, W. Su, L. Gao, S. Xia, and H. Fu, "DeepFaceDrawing: Deep generation of face images from sketches," ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH2020), vol. 39, no. 4, 72:1–72:16, 2020. Available:http://proceedings.mlr.press/v119/chen20s.html.

[8] S. Saito, T. Simon, J. Saragih, and H. Joo, Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization, 2020. arXiv:2004.00452 [cs.CV].

[9] Z. Teed and J. Deng, Raft: Recurrent all-pairs field transforms for optical flow, 2020. arXiv:2003.12039 [cs.CV].

[10] Y. Zeng, J. Fu, and H. Chao, Learning joint spatial-temporal transformations for video in-painting, 2020. arXiv:2007.10247 [cs.CV].

[Bonus 1] Z. Wan, B. Zhang, D. Chen, P. Zhang, D. Chen, J. Liao, and F. Wen, Old photo restoration via deep latent space translation, 2020. arXiv:2009.07047 [cs.CV].

[Bonus 2] Z. Ke, K. Li, Y. Zhou, Q. Wu, X. Mao, Q. Yan, and R. W. Lau, "Is a green screen really necessary for real-time portrait matting?" ArXiv, vol. abs/2011.11961, 2020.

[Bonus 3] Jason Antic, Creator of DeOldify, https://github.com/jantic/DeOldify



