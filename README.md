**distant\_viewing\_social\_media\_anniversaries**

**Repository description**

Python scripts for the analysis of visual patterns in screenshots of social media content related to the 30th anniversaries of the Fall of the Berlin Wall (2019) and German reunification (2020), and for the visualization of analysis results. These scripts were developed as part of a peer-reviewed research article submitted to *On\_Culture* Issue #20: *Celebration* under the project title *"Constructing the Anniversary: Commemorating Historical Events on Social Media and the Forms of Virtual Celebration.“* They provide a reproducible workflow for distant viewing of images and facilitate transparency in the research process. A detailed description of the scripts, their role within the broader study, and the interpretation of their outputs on the analyzed dataset will be presented in the corresponding publication.

**Scripts**

distant\_viewing\_pipeline.py\
Python-based pipeline for extracting and quantifying visual features from a collection of images. Calculates global color statistics, color histograms, face counts, and object detections using YOLOv5, saving all results in a structured CSV file for subsequent analysis.

distant\_viewing\_results\_visualization.py\
Exploratory data analysis and visualization of the distant viewing output. Produces plots that summarize brightness distribution, average color values, face occurrence rates, object detection frequencies, and RGB histograms, supporting the interpretation of visual patterns in the corpus.

**Requirements**

Python 3.8+ 
OpenCV 
NumPy 
Pandas 
face\_recognition 
PyTorch 
YOLOv5 
Matplotlib 
Seaborn

**License**

MIT License

#
