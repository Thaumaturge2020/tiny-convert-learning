# tiny-convert-learning
A tiny convert learning model 

实现：实现了一个用于语言序列对比学习的模型demo。输入两个向量化的语言序列，模型会将序列循环自动对齐并生成两个向量序列，比较他们的相似性，以此判断两个语言序列是否相同。

代码修改：在dataset_loader中进行调整即可，对于MyDataset_load_differ修改__init__与__get_item__方法
