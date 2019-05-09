使用Tensorflow實作能辨識mnist數字資料及的模型
模型的規劃:
	模型有一層隱藏層
	輸入曾有784個輸入
	隱藏曾有500個節點
	輸出則有10分類項目

優化模型是以'Adadelta'模型

訓練過程有使用:
	學習率衰減
	L2正規化
	滑動平均模型
損失函數是使用:
	交叉熵 cross entropy
訓練了31000次
使用tensorflow 的 saver儲存成jason檔