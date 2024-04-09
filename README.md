## Modelscope with IPEX-LLM

### Notebook测试环境

#### 创建并启动notebook
1. 进入Modelscope的notebook页面，选择“我的Notebook”——>PAI-DSW——>CPU环境——>启动
![1712638274904](image/README/1712638274904.png)
2. 选择查看Notebook，等待JupyterLab启动
![1712638475432](image/README/1712638475432.png)
3. 选择Terminal
![1712638629533](image/README/1712638629533.png)
#### conda虚拟环境准备
```
conda create --name ipex-llm-test python=3.9
conda activate ipex-llm-test
pip install ipex-llm[all]
pip install modelscope
pip install --upgrade transformers==4.37.0
```
如果在创建conda虚拟环境的时候遇到下面的error，请更换conda源
```
The channel is not accessible or is invalid.
```
更换方法：
```
vim /root/.condarc
```
替换为下面的conda源
```
channels:
  - defaults
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
show_channel_urls: True
```
#### 为Jupyter Notebook设置conda虚拟环境
```
# 安装ipykernel依赖
conda install -c anaconda ipykernel
# 把当前conda环境添加为Jupyter Kernel
python -m ipykernel install --user --name=ipex-llm-test
```
#### 在notebook中使用ipex-llm
下载[示例notebook文件](https://github.com/Jasonzzt/Modelscope-ipex-llm/blob/main/ipex-llm-test.ipynb)，然后上传到modelscope jupyterlab中，并选择内核ipex-llm-test后运行代码，使用其他模型时可以参考[ipex-llm repo中的example](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model)修改代码。
![1712643136692](image/README/1712643136692.png)

### 创空间测试环境
#### 进入创空间页面，使用编程式创建，填写相关信息后发布应用（选择Gradio SDK）
![1712643505924](image/README/1712643505924.png)
#### 克隆创空间
```
git lfs install
git clone http://oauth2:<your_git_token>@www.modelscope.cn/studios/<user_name>/<space_name>.git
```
#### 创建Gradio的app.py文件
单模型测试参考[ipex-llm-test](https://github.com/Jasonzzt/Modelscope-ipex-llm/blob/main/ipex-llm-test.py)，多模型可选测试参考[ipex-llm-multi-test](https://github.com/Jasonzzt/Modelscope-ipex-llm/blob/main/ipex-llm-multi-test.py).
根据需求可调整README，例如关联模型

#### 提交文件
```
git add .
git commit -m "fisrt commit"
git push
```

#### 启动并上线空间展示
提交文件后，等待5min左右代码审核
![1712644611357](image/README/1712644611357.png)

审核完成后，进入设置，选择上线空间展示以运行代码，可选设为公开空间。
![1712644679332](image/README/1712644679332.png)

可以在此查看日志和设置环境变量
![1712644781217](image/README/1712644781217.png)

#### 可用创空间
- 单模型：https://www.modelscope.cn/studios/Jasonzzt/ipex-llm-test/summary
- 多模型：https://www.modelscope.cn/studios/Jasonzzt/ipex-llm-multi-test/summary