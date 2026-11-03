import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读数据（把路径改成你本地的实际路径）
col_names = [
    "engine_id", "cycle", "setting1", "setting2", "setting3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

df_train = pd.read_csv(
    r"c:\Users\Wanghaoting\Desktop\CMAPSSData\train_FD001.txt",
    sep=r"\s+", header=None, names=col_names
)

# 2. 定义 eng1（就是这一步之前漏了）
eng1 = df_train[df_train["engine_id"] == 1]

# 3. 画多个传感器对比
for sid in ["s4", "s11", "s12"]:
    plt.figure()
    plt.plot(eng1["cycle"], eng1[sid])
    plt.title(f"engine 1 {sid}")
    plt.xlabel("cycle")
    plt.ylabel(sid)

plt.show()

# 4. 顺便把RUL标签也加上
max_cycle = df_train.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]
df_train = df_train.merge(max_cycle, on="engine_id")
df_train["rul"] = df_train["max_cycle"] - df_train["cycle"]
df_train.loc[df_train["rul"] > 125, "rul"] = 125

print("RUL列加好了，前5行：")
print(df_train[["engine_id", "cycle", "rul"]].head())
print(f"数据集形状：{df_train.shape}")


###################################################
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

# ========== 1. 先剔除无变化的传感器（C-MAPSS已知常数列）==========
drop_cols = ["s1", "s5", "s10", "s16", "s18", "s19"]
sensor_cols = [c for c in col_names if c.startswith("s") and c not in drop_cols]
setting_cols = ["setting1", "setting2", "setting3"]
feature_cols = sensor_cols + setting_cols
print(f"保留 {len(feature_cols)} 个特征列")

# ========== 2. 滑动窗口切样本 ==========
WINDOW = 30  # 每个窗口看过去30个cycle，PHM论文常用值

def build_sliding_samples(df, window=WINDOW):
    """
    对每台发动机：滑动窗口切样本
    每个窗口 = 过去window步的特征统计；标签 = 窗口最后一个cycle的rul
    """
    X_list, y_list, eng_list = [], [], []
    
    for eng_id, g in df.groupby("engine_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        feat_mat = g[feature_cols].values
        rul_arr = g["rul"].values
        
        for i in range(window - 1, len(g)):
            window_data = feat_mat[i - window + 1 : i + 1, :]  # shape: (30, n_features)
            
            # 时域统计特征：均值、标准差、max、min、偏度、峭度
            means = window_data.mean(axis=0)
            stds = window_data.std(axis=0)
            maxs = window_data.max(axis=0)
            mins = window_data.min(axis=0)
            skews = skew(window_data, axis=0)
            kurts = kurtosis(window_data, axis=0)
            
            sample = np.concatenate([means, stds, maxs, mins, skews, kurts])
            X_list.append(sample)
            y_list.append(rul_arr[i])
            eng_list.append(eng_id)
    
    return np.array(X_list), np.array(y_list), np.array(eng_list)

X_train, y_train, eng_train = build_sliding_samples(df_train)

print(f"滑动窗口样本数: {X_train.shape}")
print(f"每个样本维度: {X_train.shape[1]} (6种统计量 × {len(feature_cols)}个特征)")
print(f"RUL范围: {y_train.min():.0f} ~ {y_train.max():.0f}")


###########################################训练 XGBoost 基线模型，顺带做简单验证
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# 按发动机ID划分，不要随机shuffle（时序任务大忌，避免数据泄露）
# 取一部分发动机做验证集，保证验证发动机完全没在训练里见过
unique_eng = np.unique(eng_train)
np.random.seed(42)
np.random.shuffle(unique_eng)
train_eng = unique_eng[:90]
val_eng = unique_eng[90:]

train_mask = np.isin(eng_train, train_eng)
val_mask = np.isin(eng_train, val_eng)

X_tr, y_tr = X_train[train_mask], y_train[train_mask]
X_val, y_val = X_train[val_mask], y_train[val_mask]

print(f"训练样本：{X_tr.shape}, 验证样本：{X_val.shape}")

# 训练XGBoost回归器
model = XGBRegressor(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.08,
    random_state=42,
    objective="reg:squarederror"
)
model.fit(X_tr, y_tr)

# 预测评估
pred_tr = model.predict(X_tr)
pred_val = model.predict(X_val)

rmse_tr = math.sqrt(mean_squared_error(y_tr, pred_tr))
rmse_val = math.sqrt(mean_squared_error(y_val, pred_val))
mae_val = mean_absolute_error(y_val, pred_val)

print(f"Train RMSE: {rmse_tr:.2f}")
print(f"Val   RMSE: {rmse_val:.2f}")
print(f"Val   MAE : {mae_val:.2f}")

#######################################添加一些可视化的对比图
# 预测vs真实散点图
plt.figure(figsize=(6,6))
plt.scatter(y_val, pred_val, alpha=0.4, s=10)
plt.plot([0,125],[0,125],'r--')
plt.xlabel("True RUL")
plt.ylabel("Pred RUL")
plt.title("Validation: Predicted vs True RUL")
plt.grid(True)
plt.show()

# 特征重要性
feature_names = []
for stat in ["mean","std","max","min","skew","kurt"]:
    for s in feature_cols:
        feature_names.append(f"{s}_{stat}")
fi = pd.Series(model.feature_importances_, index=feature_names)
fi.sort_values(ascending=False).head(15).plot(kind="barh", figsize=(8,6))
plt.title("Top15 Feature Importance")
plt.show()


#####################Marcov chains' view towards this question
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# ========== 马尔可夫状态转移模型 ==========
# 5个健康状态：
# S0: RUL > 100  (健康)
# S1: 60~100     (轻度退化)
# S2: 30~60      (中度退化)
# S3: 0~30       (重度退化)
# S4: RUL = 0    (失效，吸收态)

def rul_to_state(rul):
    if rul > 100: return 0
    elif rul > 60: return 1
    elif rul > 30: return 2
    elif rul > 0: return 3
    else: return 4

n_states = 5
state_names = ["S0健康", "S1轻退", "S2中退", "S3重退", "S4失效"]

# 1. 只在train_eng上统计状态转移次数（不要用验证集数据）
trans_counts = np.zeros((n_states, n_states))
for eng_id in train_eng:
    g = df_train[df_train["engine_id"] == eng_id].sort_values("cycle")
    states = g["rul"].apply(rul_to_state).values
    for t in range(len(states) - 1):
        trans_counts[states[t], states[t+1]] += 1

# 2. 估计转移矩阵P，S4设为吸收态
P = trans_counts / trans_counts.sum(axis=1, keepdims=True)
P = np.nan_to_num(P)
P[4] = [0, 0, 0, 0, 1]

print("=== 状态转移矩阵 P ===")
print(pd.DataFrame(P, index=state_names, columns=state_names).round(3))

# 3. 吸收马尔可夫链：算每个暂态到吸收态的期望步数
# Q = 暂态之间的转移矩阵 (4x4)
# N = (I - Q)^(-1)，期望寿命 = N 每行求和
Q = P[:4, :4]
N = np.linalg.inv(np.eye(4) - Q)
expected_life = N.sum(axis=1)

print("\n=== 各状态期望剩余寿命 (cycles) ===")
for i in range(4):
    print(f"  {state_names[i]}: {expected_life[i]:.1f}")

# 4. 在验证集上评估马尔可夫模型
markov_preds, markov_trues = [], []
for eng_id in val_eng:
    g = df_train[df_train["engine_id"] == eng_id].sort_values("cycle")
    states = g["rul"].apply(rul_to_state).values
    ruls = g["rul"].values
    for t in range(len(g)):
        s = states[t]
        pred = 0 if s == 4 else expected_life[s]
        markov_preds.append(pred)
        markov_trues.append(ruls[t])

markov_preds = np.array(markov_preds)
markov_trues = np.array(markov_trues)

rmse_markov = math.sqrt(mean_squared_error(markov_trues, markov_preds))
mae_markov = mean_absolute_error(markov_trues, markov_preds)

# 5. 最终对比
print("\n=== 验证集模型对比 ===")
print(f"XGBoost Baseline:   RMSE = {rmse_val:.2f},  MAE = {mae_val:.2f}")
print(f"Markov Chain Model: RMSE = {rmse_markov:.2f},  MAE = {mae_markov:.2f}")


################################################  HMM model method
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# ======================
# 手写高斯HMM实现（Baum-Welch + Viterbi）
# ======================
class GaussianHMMFast:
    def __init__(self, n_components=5, n_iter=50, tol=1e-3, random_state=42):
        self.K = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _log_emission(self, X):
        """批量计算log高斯发射概率: X(N,D) -> (N,K)"""
        N, D = X.shape
        out = np.zeros((N, self.K))
        for k in range(self.K):
            diff = X - self.mu[k]
            out[:, k] = -0.5 * (D*np.log(2*np.pi)
                                + np.sum(2*np.log(self.sigma[k]))
                                + np.sum(diff**2 / (self.sigma[k]**2), axis=1))
        return out

    def fit(self, obs_seq_list):
        all_X = np.vstack(obs_seq_list)
        D = all_X.shape[1]
        self.mu = self.rng.normal(all_X.mean(0), all_X.std(0), (self.K, D))
        self.sigma = np.ones((self.K, D)) * all_X.std(0) + 1e-3
        self.pi = np.ones(self.K) / self.K
        self.A = np.eye(self.K) * 0.85 + np.ones((self.K, self.K)) * 0.15 / self.K
        self.A[-1, :] = 0; self.A[-1, -1] = 1.0   # 最后一个状态设为吸收态

        prev_ll = -np.inf
        for it in range(self.n_iter):
            all_gamma0 = np.zeros(self.K)
            sum_xi = np.zeros((self.K, self.K))
            sum_gamma = np.zeros(self.K)
            sum_wx = [np.zeros(D) for _ in range(self.K)]
            sum_wxx = [np.zeros(D) for _ in range(self.K)]
            total_ll = 0.0

            for obs in obs_seq_list:
                T = len(obs)
                logB = self._log_emission(obs)

                # forward
                la = np.zeros((T, self.K))
                la[0] = np.log(self.pi) + logB[0]
                for t in range(1, T):
                    s = la[t-1][:, None] + np.log(self.A)
                    la[t] = np.logaddexp.reduce(s, axis=0) + logB[t]

                # backward
                lb = np.zeros((T, self.K))
                for t in range(T-2, -1, -1):
                    s = np.log(self.A) + logB[t+1] + lb[t+1]
                    lb[t] = np.logaddexp.reduce(s, axis=1)

                # gamma
                lg = la + lb
                lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
                gamma = np.exp(lg)

                all_gamma0 += gamma[0]
                sum_gamma += gamma.sum(axis=0)

                # xi: 只对t循环，内层向量化
                for t in range(T-1):
                    xij = (la[t][:, None] + np.log(self.A)
                           + logB[t+1][None, :] + lb[t+1][None, :])
                    xij -= np.logaddexp.reduce(xij)
                    sum_xi += np.exp(xij)

                # mu/sigma充分统计量
                for k in range(self.K):
                    gk = gamma[:, k:k+1]
                    sum_wx[k] += (gk * obs).sum(axis=0)
                    sum_wxx[k] += (gk * (obs - self.mu[k])**2).sum(axis=0)

                total_ll += np.logaddexp.reduce(la[-1])

            # M step
            self.pi = all_gamma0 / len(obs_seq_list)
            self.A = sum_xi / sum_xi.sum(axis=1, keepdims=True)
            # 防止除0，加极小值保护 + 强制行归一化
            self.A = np.nan_to_num(self.A, nan=1e-6)
            self.A = self.A / self.A.sum(axis=1, keepdims=True)
            # 吸收态：最后一行固定，失效状态无法离开
            self.A[-1, :] = 0.0
            self.A[-1, -1] = 1.0
            for k in range(self.K):
                if sum_gamma[k] > 1e-8:
                    self.mu[k] = sum_wx[k] / sum_gamma[k]
                    self.sigma[k] = np.sqrt(sum_wxx[k] / sum_gamma[k]) + 1e-6

            if it > 0 and abs(total_ll - prev_ll) < self.tol:
                print(f"Baum-Welch 收敛于第 {it} 轮")
                break
            prev_ll = total_ll
        return self

    def predict(self, obs_seq_list):
        states = []
        for obs in obs_seq_list:
            logB = self._log_emission(obs)
            T = len(obs)
            delta = np.zeros((T, self.K))
            psi = np.zeros((T, self.K), dtype=int)
            delta[0] = np.log(self.pi) + logB[0]
            for t in range(1, T):
                s = delta[t-1][:, None] + np.log(self.A)
                psi[t] = s.argmax(axis=0)
                delta[t] = s.max(axis=0) + logB[t]
            z = np.zeros(T, dtype=int)
            z[-1] = delta[-1].argmax()
            for t in range(T-2, -1, -1):
                z[t] = psi[t+1, z[t+1]]
            states.extend(z.tolist())
        return np.array(states)


# ======================
# 构造观测序列（和之前逻辑一致）
# ======================
WINDOW = 30
feature_cols = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']

def build_hmm_sequences(df, eng_list, window=WINDOW):
    obs_seqs, y_seqs = [], []
    for eng_id in eng_list:
        g = df[df["engine_id"] == eng_id].sort_values("cycle").reset_index(drop=True)
        feat = g[feature_cols].values
        obs = []
        for i in range(window - 1, len(g)):
            obs.append(feat[i-window+1:i+1, :].mean(axis=0))
        obs = np.array(obs)
        obs_seqs.append(obs)
        y_seqs.append(g["rul"].values[window-1:])
    return obs_seqs, np.concatenate(y_seqs)

obs_tr_list, y_hmm_tr = build_hmm_sequences(df_train, train_eng)
obs_val_list, y_hmm_val = build_hmm_sequences(df_train, val_eng)

# 标准化 + PCA
all_tr = np.vstack(obs_tr_list)
scaler = StandardScaler()
all_tr_s = scaler.fit_transform(all_tr)
pca = PCA(n_components=5, random_state=42)
all_tr_p = pca.fit_transform(all_tr_s)

# 把每条序列转回list
ptr = 0
obs_tr_p_list = []
for seq in obs_tr_list:
    L = len(seq)
    obs_tr_p_list.append(all_tr_p[ptr:ptr+L])
    ptr += L

# 验证集变换
all_val = np.vstack(obs_val_list)
all_val_s = scaler.transform(all_val)
all_val_p = pca.transform(all_val_s)
ptr = 0
obs_val_p_list = []
for seq in obs_val_list:
    L = len(seq)
    obs_val_p_list.append(all_val_p[ptr:ptr+L])
    ptr += L

print(f"PCA累计解释方差: {pca.explained_variance_ratio_.sum():.3f}")

# 训练手写HMM
hmm = GaussianHMMFast(n_components=5, n_iter=150, random_state=42)
hmm.fit(obs_tr_p_list)

# Viterbi解码隐状态
train_states = hmm.predict(obs_tr_p_list)
val_states = hmm.predict(obs_val_p_list)

# 把隐状态按平均RUL重新排序（0健康~4失效）
state_rul = {}
for s in range(5):
    mask = train_states == s
    state_rul[s] = y_hmm_tr[mask].mean() if mask.sum()>0 else 0
sorted_health = sorted(state_rul.keys(), key=lambda x: -state_rul[x])
state_map = {orig:rank for rank,orig in enumerate(sorted_health)}

print("\n=== HMM隐状态排序（健康 -> 退化） ===")
for o in sorted_health:
    print(f"原始状态{o} → 等级{state_map[o]} | 平均RUL {state_rul[o]:.1f}")

# 用之前吸收马尔可夫算出的 expected_life 做预测
hmm_pred = np.array([
    expected_life[state_map[s]] if state_map[s] < 4 else 0
    for s in val_states
])

rmse_hmm = math.sqrt(mean_squared_error(y_hmm_val, hmm_pred))
mae_hmm = mean_absolute_error(y_hmm_val, hmm_pred)

print("\n===== 三模型汇总对比 =====")
print(f"XGBoost Baseline:       RMSE = {rmse_val:.2f}   MAE = {mae_val:.2f}")
print(f"Markov Chain (RUL离散): RMSE = {rmse_markov:.2f}   MAE = {mae_markov:.2f}")
print(f"Manual Gaussian HMM:    RMSE = {rmse_hmm:.2f}   MAE = {mae_hmm:.2f}")


##############################################visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 图1: 转移矩阵并排热力图
# ============================================================
rank_to_orig = {v: k for k, v in state_map.items()}
reordered_idx = [rank_to_orig[i] for i in range(5)]
hmm_A_reordered = hmm.A[np.ix_(reordered_idx, reordered_idx)]
hmm_A_reordered = hmm_A_reordered / hmm_A_reordered.sum(axis=1, keepdims=True)
state_labels = ["S0健康", "S1轻退", "S2中退", "S3重退", "S4失效"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(P, annot=True, fmt=".3f", cmap="Blues", ax=axes[0],
            xticklabels=state_labels, yticklabels=state_labels, vmin=0, vmax=1)
axes[0].set_title("Discrete Markov Chain\n(state = RUL threshold)")

sns.heatmap(hmm_A_reordered, annot=True, fmt=".3f", cmap="Blues", ax=axes[1],
            xticklabels=state_labels, yticklabels=state_labels, vmin=0, vmax=1)
axes[1].set_title("Gaussian HMM\n(state inferred from sensors)")

plt.tight_layout()
plt.show()

# ============================================================
# 图2: 三模型预测曲线（直接用已跑好的结果，不重算特征）
# ============================================================
target_eng = val_eng[0]
g = df_train[df_train["engine_id"] == target_eng].sort_values("cycle").reset_index(drop=True)

# --- 真实RUL ---
true_c = g["cycle"].values
true_r = g["rul"].values

# --- XGBoost：直接从已跑好的 pred_val 里取这台发动机的样本 ---
e_mask = (eng_train[val_mask] == target_eng)
xgb_p = pred_val[e_mask]
n_xgb = e_mask.sum()
xgb_c = np.arange(WINDOW, WINDOW + n_xgb)

# --- 离散马尔可夫 ---
mk_p = []
for r in g["rul"]:
    s = rul_to_state(r)
    mk_p.append(0 if s == 4 else expected_life[s])
mk_c = g["cycle"].values

# --- HMM：重新Viterbi解码这台发动机（用HMM自己的特征列）---
hmm_feature_cols = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
feat = g[hmm_feature_cols].values
obs = []
for i in range(WINDOW - 1, len(g)):
    obs.append(feat[i-WINDOW+1:i+1, :].mean(axis=0))
obs = np.array(obs)
obs_p = pca.transform(scaler.transform(obs))
states = hmm.predict([obs_p])
hmm_p = [0 if state_map[s]==4 else expected_life[state_map[s]] for s in states]
hmm_c = g["cycle"].values[WINDOW-1:]

# --- 画图 ---
plt.figure(figsize=(10, 6))
plt.plot(true_c, true_r, 'k-', linewidth=2, label="True RUL")
plt.plot(xgb_c, xgb_p, 'b--o', markersize=3, label="XGBoost")
plt.plot(mk_c, mk_p, 'g--s', markersize=3, label="Discrete Markov")
plt.plot(hmm_c, hmm_p, 'r--^', markersize=3, label="Gaussian HMM")
plt.xlabel("Cycle")
plt.ylabel("RUL")
plt.title(f"Engine #{target_eng}: RUL Prediction Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()