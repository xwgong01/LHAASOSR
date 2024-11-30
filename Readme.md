### 模拟数据

- 使用 /Simulation/MakeSimulation.ipynb 生成基于Fermi弥散背景的dataset, 需要先下载背景模型文件。推荐手动下载，也可以在运行ipynb时自动下载。

- /LHAASO_PSF 目录有KM2A的psf拟合的结果(psf_km2a.txt)，是DEC=40$^\circ$的数据.用了两个gaussian来拟合.其中第一列为能量（TeV）,第二列为sigma1的值，第三列为sigma2的值，第四列为gauss1和gauss2的振幅比

- /LHAASO_PSF 目录有一个HowToUse.ipynb展示了如何使用PSF.py 以及Plot.py,引用其中的函数可以生成PSF卷积函数。里面有些已经不再用的函数，但HowToUse是更新过的，可以参考它来看。

### 模型训练

- 模型代码保存在/SRLearning 目录。其中/SRLearning/function/ 包含了一些模型的结构实现以及一些损失函数。

- Train.ipynb 可以训练以及保存模型

- Test.ipynb 可以检查模型效果以及数据可视化

- Test_lhaaso.ipynb 是之前用五组LHAASO数据检验模型表现的，可以忽略。


