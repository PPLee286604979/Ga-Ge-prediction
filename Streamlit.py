import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

# ====================== 配置项（需根据你的实际情况修改） ======================
# 替换为你保存的镓、锗模型路径（joblib格式）
GA_MODEL_PATH = "Ga_XGB.joblib"
GE_MODEL_PATH = "Ge_XGB.joblib"


# 特征名称与对应配置（核心：按你的需求定义交互形式和取值）
FEATURE_CONFIG = {
    # 按钮选择类特征
    "pKa1-1": {
        "type": "button",
        "options": {"H₂SO₄": -8, "HCL": -2, "C₂H₂O₄": 4.76},
        "desc": "Concentration of the acid"
    },
    "pKa1-2": {
        "type": "button",
        "options": {"empty": -2, "Ca(NO₃)₂": -8, "NaNO₃": 4.76},
        "desc": "Leaching additive type"
    },
    "pKa1-3": {
        "type": "button",
        "options": {"empty": 10, "O₂": -8, "atmospheric pressure": -2, "SO₂ pressure": 4.76},
        "desc": "Pressurized gas type"
    },
    # 滑块类特征（百分比，0-100%）
    "Zn in Feed": {"type": "slider", "min": 0.0, "max": 100.0, "step": 0.01, "desc": "Zn in Feed(%)"},
    "Fe in Feed": {"type": "slider", "min": 0.0, "max": 100.0, "step": 0.01, "desc": "Fe in Feed(%)"},
    "Cu in Feed": {"type": "slider", "min": 0.0, "max": 100.0, "step": 0.01, "desc": "Cu in Feed"},
    "S in Feed": {"type": "slider", "min": 0.0, "max": 100.0, "step": 0.01, "desc": "S in Feed"},
    "Pb in Feed": {"type": "slider", "min": 0.0, "max": 100.0, "step": 0.01, "desc": "Pb in Feed"},
    "Ga in Feed": {"type": "slider", "min": 0.0, "max": 100.0, "step": 0.01, "desc": "Ga in Feed"},
    "Ge in Feed": {"type": "slider", "min": 0.0, "max": 100.0, "step": 0.01, "desc": "Ge in Feed"},
    # 数字输入类（浓度）
    "pKa1-1 Conc": {"type": "number", "step": 0.01, "desc": "pKa1-1 Conc(g/L)"},
    "pKa1-2 Conc": {"type": "number", "step": 0.01, "desc": "pKa1-2 Conc(g/L)"},
    # 滑块类特征（0-2.5）
    "pKa1-3 Conc": {"type": "slider", "min": 0.0, "max": 2.5, "step": 0.01, "desc": "pKa1-3 Conc (Mpa)"},
    # 数字输入类特征
    "Particle size": {"type": "number", "step": 0.01, "desc": "Particle size(μm)"},
    "Time": {"type": "number", "step": 0.01, "desc": "Leach time(min)"},
    "Temperature": {"type": "number", "step": 0.01, "desc": "Temperature(℃)"},
    "S/L ratio": {"type": "number", "step": 0.01, "desc": "S/L ratio"},
    "Stirring speed": {"type": "number", "step": 0.01, "desc": "Stirring speed(rpm)"}
}

# 特征顺序（与训练模型保持一致，请勿修改）
FEATURE_ORDER = [
    "Particle size", "Zn in Feed", "Fe in Feed", "Cu in Feed", "S in Feed",
    "Pb in Feed", "Ga in Feed", "Ge in Feed", "pKa1-1", "pKa1-1 Conc",
    "pKa1-2", "pKa1-2 Conc", "pKa1-3", "pKa1-3 Conc", "Time",
    "Temperature", "S/L ratio", "Stirring speed"
]

# 显示名称映射（仅用于页面展示，不影响模型输入）
DISPLAY_NAME_MAP = {
    "pKa1-1": "Concentration of the acid",
    "pKa1-2": "Leaching additive type",
    "pKa1-3": "Pressurized gas type"
}

# ====================== 页面基础设置 ======================
st.set_page_config(
    page_title="Prediction tool of galium/germanium leaching rate",
    page_icon="📊",
    layout="wide"  # 宽布局，适配三列结构
)

st.title("📊 Prediction tool of galium/germanium leaching rate")
st.divider()  # 分隔线，优化UI

# ====================== 加载预训练模型 ======================
@st.cache_resource  # 缓存模型，避免每次刷新重新加载
def load_models():
    """加载镓和锗的XGBoost模型，处理加载失败的情况"""
    try:
        # 加载镓模型
        ga_model = joblib.load(GA_MODEL_PATH)
        st.success("✅ Gallium leaching rate model was loaded successfully！")
    except Exception as e:
        ga_model = None
        st.error(f"❌ 镓浸出率模型加载失败：{str(e)}\n请检查模型路径是否正确：{GA_MODEL_PATH}")

    try:
        # 加载锗模型
        ge_model = joblib.load(GE_MODEL_PATH)
        st.success("✅ Germanium leaching rate model loading successful！")
    except Exception as e:
        ge_model = None
        st.error(f"❌ 锗浸出率模型加载失败：{str(e)}\n请检查模型路径是否正确：{GE_MODEL_PATH}")

    return ga_model, ge_model

# 执行模型加载
ga_model, ge_model = load_models()
st.divider()

# ====================== 核心布局：左-中-右三列 ======================
# 定义三列宽度比例
left_col, mid_col, right_col = st.columns([1, 2, 1.5])
feature_values = {}  # 存储特征值的字典

# ====================== 左侧列：按钮选择区 ======================
with left_col:
    st.header("📌 Type selection")
    st.markdown("---")  # 分隔线

    # pKa1-1选择
    st.markdown(f"**{FEATURE_CONFIG['pKa1-1']['desc']}**")
    pka1_1_selected = st.radio(
        label=DISPLAY_NAME_MAP["pKa1-1"],
        options=list(FEATURE_CONFIG["pKa1-1"]["options"].keys()),
        horizontal=False,
        key="pka1-1"
    )
    feature_values["pKa1-1"] = FEATURE_CONFIG["pKa1-1"]["options"][pka1_1_selected]
    st.markdown("---")

    # pKa1-2选择
    st.markdown(f"**{FEATURE_CONFIG['pKa1-2']['desc']}**")
    pka1_2_selected = st.radio(
        label=DISPLAY_NAME_MAP["pKa1-2"],
        options=list(FEATURE_CONFIG["pKa1-2"]["options"].keys()),
        horizontal=False,
        key="pka1-2"
    )
    feature_values["pKa1-2"] = FEATURE_CONFIG["pKa1-2"]["options"][pka1_2_selected]
    st.markdown("---")

    # pKa1-3选择
    st.markdown(f"**{FEATURE_CONFIG['pKa1-3']['desc']}**")
    pka1_3_selected = st.radio(
        label=DISPLAY_NAME_MAP["pKa1-3"],
        options=list(FEATURE_CONFIG["pKa1-3"]["options"].keys()),
        horizontal=False,
        key="pka1-3"
    )
    feature_values["pKa1-3"] = FEATURE_CONFIG["pKa1-3"]["options"][pka1_3_selected]

# ====================== 中间列：滑块+数字输入区 ======================
with mid_col:
    st.header("🔢 Process parameter input")
    st.markdown("---")

    # 1. 进料占比滑块
    st.subheader("📏 Feed ratio(%)")
    col_slider1, col_slider2, col_slider3 = st.columns(3)
    slider_percent_features = [
        "Zn in Feed", "Fe in Feed", "Cu in Feed", "S in Feed",
        "Pb in Feed", "Ga in Feed", "Ge in Feed"
    ]
    for idx, feat in enumerate(slider_percent_features):
        with [col_slider1, col_slider2, col_slider3][idx % 3]:
            st.markdown(f"**{FEATURE_CONFIG[feat]['desc']}**")
            val = st.slider(
                label="",
                min_value=FEATURE_CONFIG[feat]["min"],
                max_value=FEATURE_CONFIG[feat]["max"],
                step=FEATURE_CONFIG[feat]["step"],
                value=0.0,
                key=feat
            )
            feature_values[feat] = val
    st.markdown("---")

    # 2. 浓度参数
    st.subheader("🧪 Concentration parameter")
    col_conc1, col_conc2, col_conc3 = st.columns(3)
    # pKa1-1 Conc
    with col_conc1:
        st.markdown(f"**{FEATURE_CONFIG['pKa1-1 Conc']['desc']}**")
        feature_values["pKa1-1 Conc"] = st.number_input(
            label="",
            value=0.0,
            step=FEATURE_CONFIG["pKa1-1 Conc"]["step"],
            format="%.4f",
            key="pKa1-1 Conc"
        )
    # pKa1-2 Conc
    with col_conc2:
        st.markdown(f"**{FEATURE_CONFIG['pKa1-2 Conc']['desc']}**")
        feature_values["pKa1-2 Conc"] = st.number_input(
            label="",
            value=0.0,
            step=FEATURE_CONFIG["pKa1-2 Conc"]["step"],
            format="%.4f",
            key="pKa1-2 Conc"
        )
    # pKa1-3 Conc
    with col_conc3:
        st.markdown(f"**{FEATURE_CONFIG['pKa1-3 Conc']['desc']}**")
        feature_values["pKa1-3 Conc"] = st.slider(
            label="",
            min_value=FEATURE_CONFIG["pKa1-3 Conc"]["min"],
            max_value=FEATURE_CONFIG["pKa1-3 Conc"]["max"],
            step=FEATURE_CONFIG["pKa1-3 Conc"]["step"],
            value=0.0,
            key="pKa1-3 Conc"
        )
    st.markdown("---")

    # 3. 工艺参数（数字输入）
    st.subheader("⏱️ Leaching operational conditions")
    col_num1, col_num2, col_num3, col_num4, col_num5 = st.columns(5)
    number_features = [
        "Particle size", "Time", "Temperature", "S/L ratio", "Stirring speed"
    ]
    for idx, feat in enumerate(number_features):
        with [col_num1, col_num2, col_num3, col_num4, col_num5][idx]:
            st.markdown(f"**{FEATURE_CONFIG[feat]['desc']}**")
            val = st.number_input(
                label="",
                value=0.0,
                step=FEATURE_CONFIG[feat]["step"],
                format="%.4f",
                key=feat
            )
            feature_values[feat] = val
    st.markdown("---")

    # 预测按钮
    predict_btn = st.button("🚀 Start prediction", type="primary", use_container_width=True)

# ====================== 右侧列：输出展示区 ======================
with right_col:
    st.header("📈 Results Display")
    st.markdown("---")

    # 整理输入数据（匹配模型顺序）
    input_values = [feature_values[feat] for feat in FEATURE_ORDER]
    input_data = pd.DataFrame([input_values], columns=FEATURE_ORDER)

    # 预测逻辑与结果展示
    if predict_btn:
        # 先验证模型是否加载成功
        if ga_model is None and ge_model is None:
            st.warning("⚠️ 请先确保至少一个模型加载成功后再预测！")
        else:
            # 1. 输入特征值核对
            st.subheader("📋 Verify input parameters")
            display_df = input_data.copy()

            # 反向映射pKa数值到业务名称
            pka1_1_rev = {v: k for k, v in FEATURE_CONFIG["pKa1-1"]["options"].items()}
            display_df["pKa1-1"] = display_df["pKa1-1"].map(pka1_1_rev)
            pka1_2_rev = {v: k for k, v in FEATURE_CONFIG["pKa1-2"]["options"].items()}
            display_df["pKa1-2"] = display_df["pKa1-2"].map(pka1_2_rev)
            pka1_3_rev = {v: k for k, v in FEATURE_CONFIG["pKa1-3"]["options"].items()}
            display_df["pKa1-3"] = display_df["pKa1-3"].map(pka1_3_rev)

            # 列名替换为显示名称
            display_df = display_df.rename(columns=DISPLAY_NAME_MAP)

            st.dataframe(display_df, hide_index=True, use_container_width=True)
            st.markdown("---")

            # 2. 预测结果（保留2位小数）
            st.subheader("🎯 Leaching rate prediction results")
            # 镓浸出率预测
            if ga_model is not None:
                ga_pred = round(float(ga_model.predict(input_data)[0]), 2)
            else:
                ga_pred = "模型加载失败"
            # 锗浸出率预测
            if ge_model is not None:
                ge_pred = round(float(ge_model.predict(input_data)[0]), 2)
            else:
                ge_pred = "模型加载失败"

            # 卡片式展示结果
            col_ga, col_ge = st.columns(2)
            with col_ga:
                st.info("### Gallium leaching rate")
                st.metric(label="Final predicted value", value=ga_pred, delta="", delta_color="normal")
            with col_ge:
                st.success("### Germanium leaching rate")
                st.metric(label="Final predicted value", value=ge_pred, delta="", delta_color="normal")
    else:
        # 未点击预测时的提示
        st.info("💡 Click the button to view the results")

# ====================== 底部提示 ======================
st.divider()
st.caption(
    "💡 Point out：\n1. The input eigenvalue range matches the data used during model training；\n2. Replace the model path with your locally saved joblib format model path；\n3. All input parameters should be filled in according to actual process conditions。")
