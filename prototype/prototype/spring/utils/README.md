## profiling用法

首先需要安装pyprof

```bash
git clone https://github.com/NVIDIA/PyProf.git
cd PyProf
rm -rf .git
pip install .
```

然后修改训练代码，在代码中加上

```python
import torch.cuda.profiler as profiler
import pyprof
pyprof.init()
```

在训练代码部分，做类似如下修改

```python
iters = 500
iter_to_capture = 100

# Define network, loss function, optimizer etc.

# PyTorch NVTX context manager
with torch.autograd.profiler.emit_nvtx():

    for iter in range(iters):

        if iter == iter_to_capture:
            profiler.start()

        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if iter == iter_to_capture:
            profiler.stop()
```

在集群上的启动脚本，以prototype为例，简单来说，就是在之前的脚本前面加上`nvprof -f -o prof_out/net_%p.sql --profile-from-start off`

```bash
mkdir -p prof_out
PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
nvprof -f -o prof_out/net_%p.sql --profile-from-start off python -u -m prototype.solver.vit_solver --solver ViTSolver  --config config.yaml  # --evaluate
```

生成net.sql之后，可以解析一下

```bash
srun -p Test python -m prototype.spring.utils.profiling --input prof_out/net_53780.sql --output prof_out
```
