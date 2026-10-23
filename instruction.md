# 热电子结构的无标签路径反演

实现 `/app/src/thermal_path/__init__.py` 中的 `reconstruct`；可以在该包内增设模块。研究模型是基于真实 PySCF 2.14.0 GHF 的作者扩展，输入是合成的一体观测数据。不是上游已知缺陷，也不假称实验测量。已安装的 NumPy、SciPy 和 PySCF 均可使用。

## 科学对象

`mf` 是未运行或已运行的原生分子 `pyscf.scf.ghf.GHF`，不含 DFT 或密度拟合改写。其 AO 自旋块次序沿用 PySCF。令 S=mf.get_ovlp()，h0=mf.get_hcore()，G(D)=mf.get_veff(mf.mol,D,hermi=1)，Enuc=mf.energy_nuc()。使用完整的复数自旋密度与 GHF 库仑/交换作用；不假设自旋共线、矩阵为实数或 AO 正交。

D 是 AO 一体密度，P=S^(1/2) D S^(1/2) 为正交度量中的占据矩阵。物理域为 0<P<I。这里固定的是平均粒子数，不是整数粒子数投影后的多体正则系综。熵为高斯费米态的 s(D)=-Tr[P log(P)+(I-P) log(I-P)]，无量纲，kB=1。

`generators` 给出 m 个无量纲 Hermitian AO 算符 V_i。未知外场系数 w_i 与温度宽度 T 均以 Hartree 表示，T 表示 kB 乘绝对温度。`constraints` 给出 r 个固定 Hermitian 算符 A_a，`targets` 给出 Tr(A_a D)=c_a；第一个 A 为 S，第一个 c 为 mf.mol.nelectron。

给定 w、T，自洽电子态定义为在上述物理域和固定约束下使以下自由能最小的 D：

E(D,w)=Enuc+Tr[(h0+sum_i w_i V_i)D]+(coupling/2)Tr[D G(D)]；
F(D,w,T)=E(D,w)-T s(D)。

`coupling` 是给定的无量纲两电子作用比例。约束乘子 lambda 的符号约定由变分拉格朗日量 F-sum_a lambda_a(Tr(A_a D)-c_a) 决定。

每条记录给出 m 个观测槽 `values` 和一个熵 `entropy`。指定的 `unlabeled_indices` 中，通道标签在每条记录里分别未知；其余槽标签可靠。输出排列 `mapping` 的定义是：若 U 为该索引序列，通道 U[j] 的期望值是原始 `values[U[mapping[j]]]`。同样的排列应用于该记录的观测方向。一个可行对应必须给出物理密度，并同时符合全部标记观测、固定约束及给定熵。

全部标记观测与固定约束共同张成 Hermitian 密度空间，因此标签已知时密度唯一；标签未知时可有多个物理候选。`candidate_counts[j]` 是第 j 条记录所有可行排列的数量，按排列计数而非按数值近似相同的密度合并。

## 全路径选择

记参数向量 z_j=(w_j,T_j)，给定初始锚点 z_0。对于目前收到的完整前缀，在所有可行通道对应、自洽密度和温度区间内的参数代表元中，联合最小化

J=sum_(j=1..N) ||z_j-z_(j-1)||_2^2。

这是使用上述固定参数单位的表示约定，不宣称它等于真实温度，也不宣称是材料的物理滞后。所有帧共同参与最小化；追加观测后，既有帧的最优对应和参数代表元可以改变。返回整个当前前缀的重建结果。

`joint_gauge_projector` 是该帧固定对应和固定观测下，局部不可辨识参数方向在 (w,T) 欧氏坐标中的正交投影矩阵；包含冗余外场以及温度与外场之间的不可辨识性。它描述观测约束下的完整局部参数族，不是全路径正则化后的 Hessian 逆。

## 接口与输入

```python
reconstruct(mf, generators, constraints, targets, records,
            unlabeled_indices=(), initial_anchor=None,
            initial_directions=None, checkpoint=None,
            temperature_interval=(0.4, 1.6), coupling=1.0)
```

矩阵输入为 NumPy 数组或等价嵌套序列；形状分别是 (m,n,n)、(r,n,n)、(r,)。矩阵可以为复数。`records` 是有序 list，每项包含 `values`（m 个实数）、`entropy`（实数）及可选 `directions`（q,m+1）。方向最后一列是熵的变化，其余列是原始槽的变化。缺少 directions 表示该记录的 q 个方向均为零。

初次调用 `initial_anchor` 为 (m+1,)，缺省是 (0,...,0,1)。`initial_directions` 为 (q,m+1)，含每个查询下锚点的变化；缺省由首条记录方向数确定 q，锚点方向取零，无记录则 q=0。一个会话内 q 固定，记录方向不能改变固定约束、原子几何、两电子作用、算符或温度区间。

响应是联合最优解对上述输入方向的一阶导数，必须考虑所有帧及给定锚点的变化。全局对应在查询的局部邻域保持不变；温度边界的活动状态也保持不变。可行方向满足密度约束和熵的一阶相容性。未来帧的查询可以改变过去帧的参数响应，而过去帧的物理密度仍由该帧的完整观测决定。

支持域：2<=n<=8，1<=r<=4，m<=n*n+4，0<=q<=4，整个会话最多32条记录，最多4个未标记通道。温度区间 0.3<=Tmin<Tmax<=3，0<coupling<=1。固定约束独立，标记观测和约束满秩，其非零奇异值条件数不超过10^6。可行占据位于 [0.1,0.9]，所有物理候选的熵与数据相符；不可行排列与物理域、线性相容或熵条件存在明确数值间隔。每帧至少一个、最多四个可行排列。完整全路径对应具有唯一最优解，其离散最优值与其他对应分离；连续问题满足所请求导数的固定秩及严格边界互补条件。约束的零模不因查询改变。

在正交 Hermitian 密度的 Frobenius 度量下，二电子能量的线性梯度算子 K 满足 4*Tmin+lambda_min(K)>10^-4，保证整个温度区间自由能的一致严格凸性。K 已包含 coupling。此条件不把 AO 度量当成单位阵。

## 输出

返回 dict，恰含 `frames`、`objective`、`objective_response`、`candidate_counts`、`checkpoint`。

`frames` 是整个前缀的有序 list；每项恰含：

- `mapping`：U 的排列，Python 整数 list，U 为空时是空 list。
- `density`：(n,n) 复数 AO 密度。
- `coefficients`：(m,) 外场系数，`temperature`：实数 T。
- `constraint_multipliers`：(r,) 约束乘子，符号如上。
- `joint_gauge_projector`：(m+1,m+1) 实对称正交投影。
- `entropy`、`energy`、`free_energy`：上面定义的实数。
- `parameter_response`：(q,m+1)，`density_response`：(q,n,n)。q=0 时保留零长度首维。
- `temperature_bound_multiplier`：有符号的温度边界乘子，即 J 对该帧 T 的偏导；内部为0，下界非负，上界非正。J 的定义没有乘1/2。

`objective` 是 J，`objective_response` 为 (q,)，`candidate_counts` 是每条记录的整数 list。数值输出必须有限；允许 ndarray 或可转换成规定形状的数值序列。除 checkpoint 外不接受编码过的数组字符串。数值验收使用 atol=5e-7、rtol=3e-6；响应使用 atol=3e-6、rtol=1e-5，精确键、形状、排列与计数独立检查。

## 检查点与失败行为

`checkpoint` 必须是只含 JSON 数据的 dict，经标准 JSON 序列化/反序列化后可续接；不能依赖进程内对象、临时文件或 pickle。检查点信封恰含 `schema_version`（1）、`context`（非空科学上下文标识字符串）、`payload`（JSON dict）、`sha256`。摘要为前面三个字段组成的对象在 json.dumps(sort_keys=True,separators=(',', ':'),ensure_ascii=True,allow_nan=False) 下的 ASCII 字节 SHA-256。payload 内部表示自定，必须足以恢复相同完整前缀及其查询。

续接时，初始锚点及初始方向从 checkpoint 取得，调用方不再传 initial_anchor 或 initial_directions。相同上下文的整段调用与任意合法分批续接，必须返回相同的完整前缀最优结果和响应；检查点字节无需与整段结果相同。科学上下文改变时必须拒收旧检查点；上下文包含原生 S、h0、两电子积分、Enuc、算符、固定目标、未标记索引、coupling 和温度区间。没有记录时可建立或复读检查点，结果为空前缀 J=0，方向形状保留。

损坏或版本不支持的检查点、上下文不符、非法输入形状、非有限输入、记录无可行对应或不相容查询，抛出 ValueError；其他公开支持域条件由调用方保证。不得修改 mf、算符、输入记录或传入的 checkpoint。批次后部拒收时，传入的 checkpoint 和此前返回的对象保持原值。

## 执行边界

仅 `/app/src` 的候选源码随交付转入验收环境。验收不提供答案或参考实现，候选在独立进程中执行。每次接口调用最多180秒，2 CPU、4 GiB；安装和模型通信不是候选科学计算时限。可新增辅助源码，禁止依靠修改测试、读取作者材料、输出预置答案或更改运行环境获得分数。
