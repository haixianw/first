import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class SDEParameterEstimator:
    """
    使用扩展卡尔曼滤波(EKF)和最大似然估计对随机微分方程组进行参数估计。
    
    该类处理以下方程组:
    δS_t = r S_t δt + β L_t S_t δW_t^γ + σ_S S_t δW_t^S
    δL_t = α(θ̄ - L_t)δt + σ_L δW_t^L
    
    其中:
    - S_t: 股票价格 (可观测)
    - L_t: 市场流动性 (潜变量/不可观测)
    """
    
    def __init__(self, stock_prices, dt):
        """
        初始化估计器
        
        参数:
        - stock_prices: 股票价格时间序列数据 (numpy array)
        - dt: 时间步长
        """
        self.stock_prices = stock_prices
        self.dt = dt
        self.n = len(stock_prices)
        
    def simulate_sde(self, params, n_steps, dt, S0=100, L0=1.0, random_seed=None):
        """
        使用Euler-Maruyama方法模拟SDE
        
        参数:
        - params: (r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3)
        - n_steps: 模拟步数
        - dt: 时间步长
        - S0: 初始股票价格
        - L0: 初始流动性
        
        返回:
        - S: 股票价格序列
        - L: 流动性序列
        """
        r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3 = params
        
        # 设置随机数种子使结果可复现
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 初始化
        S = np.zeros(n_steps)
        L = np.zeros(n_steps)
        S[0] = S0
        L[0] = L0
        
        # 生成相关的布朗运动
        # 创建协方差矩阵
        cov_matrix = np.array([
            [1, rho1, rho2],
            [rho1, 1, rho3],
            [rho2, rho3, 1]
        ])
        
        # 生成相关的随机数
        random_numbers = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=cov_matrix,
            size=n_steps-1
        )
        
        dW_gamma = random_numbers[:, 0] * np.sqrt(dt)
        dW_S = random_numbers[:, 1] * np.sqrt(dt)
        dW_L = random_numbers[:, 2] * np.sqrt(dt)
        
        # 欧拉-马鲁亚马方法模拟SDE
        for t in range(n_steps-1):
            # 更新股票价格
            drift_S = r * S[t] * dt
            diffusion1_S = beta * L[t] * S[t] * dW_gamma[t]
            diffusion2_S = sigma_S * S[t] * dW_S[t]
            S[t+1] = S[t] + drift_S + diffusion1_S + diffusion2_S
            
            # 更新流动性
            drift_L = alpha * (theta_bar - L[t]) * dt
            diffusion_L = sigma_L * dW_L[t]
            L[t+1] = L[t] + drift_L + diffusion_L
            
            # 确保S和L保持正值
            S[t+1] = max(S[t+1], 1e-10)
            L[t+1] = max(L[t+1], 1e-10)
        
        return S, L
    
    def extended_kalman_filter(self, params, S_observed):
        """
        使用扩展卡尔曼滤波估计潜变量L
        
        参数:
        - params: (r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3)
        - S_observed: 观测到的股票价格
        
        返回:
        - L_estimated: 估计的流动性序列
        - log_likelihood: 对数似然值
        """
        r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3 = params
        dt = self.dt
        n = len(S_observed)
        
        # 初始化
        L_predicted = np.zeros(n)
        L_updated = np.zeros(n)
        P_predicted = np.zeros(n)
        P_updated = np.zeros(n)
        log_likelihood = 0
        
        # 初始状态估计
        L_updated[0] = theta_bar  # 初始估计为长期均值
        P_updated[0] = sigma_L**2 / (2 * alpha)  # 稳态协方差
        
        for t in range(1, n):
            # 预测步骤
            L_predicted[t] = L_updated[t-1] + alpha * (theta_bar - L_updated[t-1]) * dt
            P_predicted[t] = P_updated[t-1] + dt * (
                -2 * alpha * P_updated[t-1] + sigma_L**2
            )
            
            # 计算观测噪声方差
            S_prev = S_observed[t-1]
            dS_observed = S_observed[t] - S_prev
            expected_dS = r * S_prev * dt
            
            # 计算卡尔曼增益
            H = beta * S_prev * np.sqrt(dt)  # 观测方程的雅可比
            R = (sigma_S * S_prev * np.sqrt(dt))**2  # 观测噪声方差
            K = P_predicted[t] * H / (H**2 * P_predicted[t] + R)
            
            # 更新步骤
            innovation = dS_observed - expected_dS
            L_updated[t] = L_predicted[t] + K * innovation
            P_updated[t] = (1 - K * H) * P_predicted[t]
            
            # 计算对数似然
            innovation_variance = H**2 * P_predicted[t] + R
            log_likelihood += -0.5 * (np.log(2 * np.pi * innovation_variance) + 
                                    innovation**2 / innovation_variance)
        
        return L_updated, log_likelihood
    
    def negative_log_likelihood(self, params):
        """
        计算负对数似然函数（用于最小化）
        
        参数:
        - params: 参数向量 (r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3)
        
        返回:
        - negative_ll: 负对数似然值
        """
        # 检查参数约束
        r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3 = params
        
        # 参数约束检查
        if (alpha <= 0 or sigma_S <= 0 or sigma_L <= 0 or 
            theta_bar <= 0 or
            abs(rho1) >= 1 or abs(rho2) >= 1 or abs(rho3) >= 1):
            return 1e10  # 返回一个很大的数表示无效的参数
        
        # 计算对数似然
        _, log_likelihood = self.extended_kalman_filter(params, self.stock_prices)
        
        return -log_likelihood
    
    def estimate_parameters(self, initial_params, bounds=None):
        """
        使用最大似然估计参数
        
        参数:
        - initial_params: 初始参数估计 (r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3)
        - bounds: 参数的边界约束(可选)
        
        返回:
        - optimal_params: 估计的最优参数
        - L_estimated: 估计的流动性序列
        """
        if bounds is None:
            # 默认的参数边界
            bounds = [
                (-0.5, 0.5),      # r
                (1e-6, 10),       # alpha
                (-10, 10),        # beta
                (1e-6, 10),       # theta_bar
                (1e-6, 2),        # sigma_S
                (1e-6, 2),        # sigma_L
                (-0.99, 0.99),    # rho1
                (-0.99, 0.99),    # rho2
                (-0.99, 0.99)     # rho3
            ]
        
        # 最小化负对数似然
        result = minimize(
            self.negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        optimal_params = result.x
        
        # 使用最优参数估计流动性
        L_estimated, _ = self.extended_kalman_filter(optimal_params, self.stock_prices)
        
        return optimal_params, L_estimated
    
    def plot_results(self, S_observed, L_estimated, params, title="SDE Parameter Estimation Results"):
        """
        绘制结果
        """
        r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3 = params
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        time = np.arange(len(S_observed)) * self.dt
        
        # 绘制股票价格
        ax1.plot(time, S_observed, 'b-', label='Observed Stock Price')
        ax1.set_ylabel('Stock Price (S)')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制估计的流动性
        ax2.plot(time, L_estimated, 'r-', label='Estimated Liquidity (L)')
        ax2.set_ylabel('Liquidity (L)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        fig.suptitle(title)
        
        # 显示估计的参数
        param_text = (
            f"Estimated Parameters:\n"
            f"r = {r:.4f}, α = {alpha:.4f}, β = {beta:.4f}, θ̄ = {theta_bar:.4f}\n"
            f"σ_S = {sigma_S:.4f}, σ_L = {sigma_L:.4f}\n"
            f"ρ1 = {rho1:.4f}, ρ2 = {rho2:.4f}, ρ3 = {rho3:.4f}"
        )
        fig.text(0.5, 0.01, param_text, ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.show()


# 演示如何使用此类的示例代码
if __name__ == "__main__":
    # 设置真实参数
    true_params = (
        0.05,    # r: 无风险利率
        0.5,     # alpha: 均值回归速度
        0.1,     # beta: 流动性影响系数
        1.0,     # theta_bar: 长期平均流动性
        0.2,     # sigma_S: 股票价格波动率
        0.1,     # sigma_L: 流动性波动率
        0.3,     # rho1: 相关系数1
        0.2,     # rho2: 相关系数2
        0.1      # rho3: 相关系数3
    )
    
    # 模拟参数
    dt = 1/252    # 日度数据(年化)
    n_years = 2
    n_steps = int(n_years / dt)
    
    # 使用真实参数生成模拟数据
    np.random.seed(42)  # 设置随机种子以复现结果
    
    # 创建临时估计器对象用于模拟
    temp_estimator = SDEParameterEstimator(None, dt)
    S_simulated, L_simulated = temp_estimator.simulate_sde(true_params, n_steps, dt, S0=100, L0=1.0)
    
    # 创建估计器
    estimator = SDEParameterEstimator(S_simulated, dt)
    
    # 设置初始参数猜测(偏离真实值)
    initial_params = (
        0.03,    # r
        0.7,     # alpha
        0.2,     # beta
        0.8,     # theta_bar
        0.15,    # sigma_S
        0.08,    # sigma_L
        0.2,     # rho1
        0.1,     # rho2
        0.0      # rho3
    )
    
    # 估计参数
    estimated_params, L_estimated = estimator.estimate_parameters(initial_params)
    
    print("真实参数:", true_params)
    print("估计参数:", estimated_params)
    
    # 计算估计误差
    error = np.abs(np.array(estimated_params) - np.array(true_params))
    print("参数估计误差:", error)
    
    # 绘制结果
    estimator.plot_results(S_simulated, L_estimated, estimated_params, 
                           title="SDE Parameter Estimation: Simulated vs Estimated")
    
    # 绘制真实与估计的流动性对比
    plt.figure(figsize=(10, 6))
    time = np.arange(len(S_simulated)) * dt
    plt.plot(time, L_simulated, 'b-', label='True Liquidity')
    plt.plot(time, L_estimated, 'r--', label='Estimated Liquidity')
    plt.xlabel('Time')
    plt.ylabel('Liquidity (L)')
    plt.title('True vs Estimated Liquidity')
    plt.legend()
    plt.grid(True)
    plt.show() 