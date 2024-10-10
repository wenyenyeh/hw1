import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 全局的網頁標題與描述
st.set_page_config(page_title="Linear Regression Demo", layout="wide")
st.title("🌟 Linear Regression Demo with Adjustable Parameters 🌟")
st.markdown("Use the sliders on the left to adjust the parameters and visualize the impact on the regression model in real time.")

# 使用兩欄布局
col1, col2 = st.columns([1, 2])

# 在第一欄中設置參數
with col1:
    st.markdown("### 🔧 **Input Parameters**")
    st.markdown("---")  # 分隔線
    slope = st.slider('Select slope (a)', -100.0, 100.0, -50.0)
    noise_scale = st.slider('Select noise scale (c)', 0.0, 100.0, 5.0)
    n_points = st.slider('Select number of points (n)', 10, 500, 100)

# 生成數據
np.random.seed(42)
X = 2 * np.random.rand(n_points, 1)
y = slope * X + np.random.randn(n_points, 1) * noise_scale

# 建立線性回歸模型並進行擬合
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 計算評估指標
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 在第二欄中顯示結果和圖表
with col2:
    st.markdown("### 📊 **Model Evaluation Metrics**")
    st.markdown("---")  # 分隔線
    st.markdown(f"**Mean Squared Error (MSE):** ` {mse:.2f} `")
    st.markdown(f"**R-squared:** ` {r2:.2f} `")

    # 繪製散點圖與回歸線
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Actual Points', alpha=0.6)
    ax.plot(X, y_pred, color='red', label='Regression Line', linewidth=2)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.set_title("Actual vs Predicted with Regression Line")
    ax.legend()

    # 顯示圖表
    st.pyplot(fig)

# 增加底部的註解或結尾
st.markdown("---")
st.markdown("##### Created by Angela Yeh | Powered by Streamlit & Scikit-learn")
