# TEVMapCutter
## 1、Feature
• Prediction tools focused on determining whether TEV interacts with peptide substrates.<br>
• Utilize advanced deep learning models to accurately predict the interactions between tobacco etch virus (TEV) proteases and peptide substrates.<br>
• Support loading embedding vectors of peptides and proteases from CSV files and perform zero-padding operations on the data to meet different input requirements.<br>
• Provide a heatmap plotting function to intuitively display contact maps and help users better understand the interaction scenarios.<br>
• Technical path:<br>
 ![image]([https://datashare.biochem.mpg.de/s/ac9ufZ0NB2IrkZL](https://github.com/2053798680wang/TEVMapCutter/blob/main/%E5%85%A8%E6%96%87%E6%A1%86%E6%9E%B6.png))


## 2、<span style="background-color: yellow;">Train MPCutter</span> 
### Required configuration and environment
||   Python=3.7   ||   Pytorch=2.1   ||   scikit_learn=1.3.0   ||   PST_t6  [link](https://datashare.biochem.mpg.de/s/ac9ufZ0NB2IrkZL)   ||	<br><br>
•	Train_model contains the code for model training. <br>
•	best_model is the one we attempted to adopt in this experiment. <br>
•	contact_map is s the extraction process of the residue contact map. <br>
•	embedding is the embedding vector extracted by the trained model. <br>
