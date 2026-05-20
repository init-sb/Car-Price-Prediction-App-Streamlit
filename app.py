import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding: 2rem 3rem; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
[data-testid="stSidebar"] { background: #111 !important; border-right: 1px solid #222; }
[data-testid="stSidebar"] * { color: #ccc !important; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #ffffff; line-height: 1.1; margin-bottom: 0.3rem; }
.hero-sub { font-size: 0.95rem; color: #666; margin-bottom: 2rem; }
.accent { color: #e8c547; }
.card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 16px; padding: 1.8rem; margin-bottom: 1rem; }
.card-title { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #e8c547; margin-bottom: 1.2rem; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.metric-card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 14px; padding: 1.5rem; text-align: center; }
.metric-card.highlight { border-color: #e8c547; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #e8c547; }
.metric-lbl { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
.metric-desc { font-size: 0.8rem; color: #444; margin-top: 0.5rem; }
.result-box { background: linear-gradient(135deg, #1a1a1a 0%, #1f1f1f 100%); border: 1px solid #e8c547; border-radius: 16px; padding: 2.5rem; text-align: center; margin-top: 1.5rem; }
.result-label { font-size: 0.8rem; color: #666; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem; }
.result-price { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; color: #e8c547; line-height: 1; }
.result-range { font-size: 0.82rem; color: #444; margin-top: 0.8rem; }
.result-badges { display: flex; gap: 0.8rem; justify-content: center; margin-top: 1.2rem; flex-wrap: wrap; }
.badge { background: #111; border: 1px solid #2a2a2a; border-radius: 8px; padding: 0.4rem 1rem; font-size: 0.78rem; color: #aaa; }
.feat-row { margin-bottom: 0.9rem; }
.feat-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #aaa; margin-bottom: 0.3rem; }
.feat-bar-bg { background: #1a1a1a; border-radius: 999px; height: 8px; overflow: hidden; }
.feat-bar-fill { height: 8px; border-radius: 999px; background: linear-gradient(90deg, #e8c547, #f5a623); }
.desc-block { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 14px; padding: 1.5rem 2rem; margin-bottom: 1rem; }
.desc-block h4 { font-family: 'Syne', sans-serif; color: #e8c547; font-size: 0.95rem; margin-bottom: 0.5rem; }
.desc-block p, .desc-block li { color: #888; font-size: 0.88rem; line-height: 1.7; }
.stButton > button { width: 100%; background: #e8c547; color: #0f0f0f; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; border: none; border-radius: 12px; padding: 0.85rem 2rem; letter-spacing: 0.05em; }
.stButton > button:hover { opacity: 0.85; color: #0f0f0f; }
.stSelectbox label, .stNumberInput label, .stSlider label { color: #aaa !important; font-size: 0.85rem !important; }
.stSelectbox > div > div { background: #111 !important; border: 1px solid #2a2a2a !important; border-radius: 10px !important; color: #fff !important; }
.stNumberInput > div > div > input { background: #111 !important; border: 1px solid #2a2a2a !important; border-radius: 10px !important; color: #fff !important; }
.divider { border: none; border-top: 1px solid #1f1f1f; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('car_price_predict.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 2rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800; color: #fff;'>🚗 CarPredict</div>
        <div style='font-size: 0.75rem; color: #444; margin-top: 0.2rem;'>ML Price Estimator</div>
    </div>
    """, unsafe_allow_html=True)
    menu = st.radio("Navigation", ["🔍 Predict Price", "📖 Description", "📉 EDA", "📊 Model Evaluation"], label_visibility="collapsed")
    st.markdown("""
    <div style='margin-top: 3rem; font-size: 0.72rem; color: #333; line-height: 1.8;'>
        Random Forest Regression<br>R² Test: 0.804 · MAPE: 7.98%<br>Dataset: 1,000 cars
    </div>
    """, unsafe_allow_html=True)

# ── PAGE 1: PREDICT ──────────────────────────────────────────
if menu == "🔍 Predict Price":
    st.markdown('<div class="hero-title">Predict Your<br><span class="accent">Car Price</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Fill in the details below and get an instant price estimate</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="card-title">🚘 Car Identity</div>', unsafe_allow_html=True)
        make         = st.selectbox('Brand', ['Honda', 'Ford', 'BMW', 'Audi', 'Toyota'])
        fuel_type    = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric'])
        transmission = st.selectbox('Transmission', ['Automatic', 'Manual'])
        year         = st.slider('Year', min_value=2000, max_value=2021, value=2015)
    with col2:
        st.markdown('<div class="card-title">⚙️ Specifications</div>', unsafe_allow_html=True)
        engine_size = st.slider('Engine Size (L)', min_value=1.0, max_value=4.5, value=2.0, step=0.1)
        mileage     = st.number_input('Mileage (km)', min_value=56, max_value=199867, value=50000, step=1000)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button('🔍 Estimate Price Now')

    if predict_btn:
        data_baru = pd.DataFrame({
            'Year': [year], 'Engine Size': [engine_size], 'Mileage': [mileage],
            'Make': [make], 'Fuel Type': [fuel_type], 'Transmission': [transmission]
        })
        prediksi = model.predict(data_baru)[0]
        mae = 1868
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Price</div>
            <div class="result-price">${prediksi:,.0f}</div>
            <div class="result-range">Likely range: <strong style="color:#aaa">${prediksi-mae:,.0f}</strong> – <strong style="color:#aaa">${prediksi+mae:,.0f}</strong> &nbsp;·&nbsp; ±MAE $1,868</div>
            <div class="result-badges">
                <span class="badge">🏷️ {make}</span>
                <span class="badge">📅 {year}</span>
                <span class="badge">⛽ {fuel_type}</span>
                <span class="badge">⚙️ {transmission}</span>
                <span class="badge">🔧 {engine_size}L</span>
                <span class="badge">🛣️ {mileage:,} km</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── PAGE 2: EVALUATION ───────────────────────────────────────
elif menu == "📊 Model Evaluation":
    st.markdown('<div class="hero-title">Model<br><span class="accent">Evaluation</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Performance metrics and insights from the trained Random Forest model</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card highlight">
            <div class="metric-val">0.804</div>
            <div class="metric-lbl">R² Score (Test)</div>
            <div class="metric-desc">Model explains 80.4% of price variance</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">7.98%</div>
            <div class="metric-lbl">MAPE</div>
            <div class="metric-desc">Average prediction error in percentage</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">$1,868</div>
            <div class="metric-lbl">MAE</div>
            <div class="metric-desc">Average absolute price deviation</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">$2,315</div>
            <div class="metric-lbl">RMSE</div>
            <div class="metric-desc">Penalizes large prediction errors</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card-title">📈 Train vs Test Score</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="desc-block">
            <div style="margin-bottom:1.2rem;">
                <div class="feat-label"><span>Train R²</span><span style="color:#e8c547">0.906</span></div>
                <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:90.6%"></div></div>
            </div>
            <div>
                <div class="feat-label"><span>Test R²</span><span style="color:#e8c547">0.804</span></div>
                <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:80.4%"></div></div>
            </div>
            <p style="margin-top:1rem; font-size:0.8rem; color:#444;">Gap ~10% — model generalizes well without severe overfitting.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-title" style="margin-top:1.5rem">⚙️ Best Parameters</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="desc-block">
            <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #2a2a2a;"><td style="padding:0.5rem 0; color:#666;">n_estimators</td><td style="color:#e8c547; text-align:right;">100</td></tr>
                <tr style="border-bottom:1px solid #2a2a2a;"><td style="padding:0.5rem 0; color:#666;">max_depth</td><td style="color:#e8c547; text-align:right;">None</td></tr>
                <tr style="border-bottom:1px solid #2a2a2a;"><td style="padding:0.5rem 0; color:#666;">min_samples_leaf</td><td style="color:#e8c547; text-align:right;">4</td></tr>
                <tr><td style="padding:0.5rem 0; color:#666;">min_samples_split</td><td style="color:#e8c547; text-align:right;">10</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-title">🏆 Feature Importance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="desc-block">
            <div class="feat-row"><div class="feat-label"><span>Year</span><span style="color:#e8c547">High (corr: 0.61)</span></div><div class="feat-bar-bg"><div class="feat-bar-fill" style="width:85%"></div></div></div>
            <div class="feat-row"><div class="feat-label"><span>Mileage</span><span style="color:#e8c547">High (corr: -0.56)</span></div><div class="feat-bar-bg"><div class="feat-bar-fill" style="width:78%"></div></div></div>
            <div class="feat-row"><div class="feat-label"><span>Engine Size</span><span style="color:#e8c547">Medium (corr: 0.38)</span></div><div class="feat-bar-bg"><div class="feat-bar-fill" style="width:52%"></div></div></div>
            <div class="feat-row"><div class="feat-label"><span>Make</span><span style="color:#e8c547">Low–Medium</span></div><div class="feat-bar-bg"><div class="feat-bar-fill" style="width:35%"></div></div></div>
            <div class="feat-row"><div class="feat-label"><span>Fuel Type</span><span style="color:#e8c547">Low</span></div><div class="feat-bar-bg"><div class="feat-bar-fill" style="width:20%"></div></div></div>
            <div class="feat-row"><div class="feat-label"><span>Transmission</span><span style="color:#e8c547">Low</span></div><div class="feat-bar-bg"><div class="feat-bar-fill" style="width:15%"></div></div></div>
        </div>
        """, unsafe_allow_html=True)

# ── PAGE 3: EDA ──────────────────────────────────────────────
elif menu == "📉 EDA":
    st.markdown('<div class="hero-title">Exploratory<br><span class="accent">Data Analysis</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Visual insights from the dataset before model training</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Scatter Plot ─────────────────────────────────────────
    st.markdown('<div class="card-title">🔵 Scatter Plot — Price vs Mileage</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1], gap="large")
    with col1:
        st.image('images/scatter_plot.png', use_column_width=True)
    with col2:
        st.markdown("""
        <div class="desc-block" style="height:100%">
            <h4>📌 What This Shows</h4>
            <p>This scatter plot displays the relationship between <strong style="color:#fff">Mileage</strong> and <strong style="color:#fff">Price</strong> for every car in the dataset.</p>
            <br>
            <h4>💡 Key Insights</h4>
            <ul>
                <li>A clear <strong style="color:#fff">negative trend</strong> — higher mileage leads to lower price</li>
                <li>Mileage vs Price correlation = <strong style="color:#e8c547">-0.56</strong> (moderately strong)</li>
                <li>Cars with low mileage (0–25,000 km) tend to be priced at <strong style="color:#fff">$30,000–$42,000</strong></li>
                <li>Cars with high mileage (175,000+ km) drop to <strong style="color:#fff">$10,000–$20,000</strong></li>
                <li>A few <strong style="color:#fff">outliers</strong> exist at high mileage with unexpectedly high prices</li>
            </ul>
            <br>
            <h4>🤖 Model Implication</h4>
            <p>Mileage is one of the <strong style="color:#e8c547">most influential</strong> features for price prediction — ranked second after Year.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Histogram ────────────────────────────────────────────
    st.markdown('<div class="card-title">📊 Histogram — Price Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.4], gap="large")
    with col1:
        st.markdown("""
        <div class="desc-block">
            <h4>📌 What This Shows</h4>
            <p>This histogram shows the <strong style="color:#fff">price distribution</strong> across all cars — how many cars fall within each price range.</p>
            <br>
            <h4>💡 Key Insights</h4>
            <ul>
                <li>Price distribution is close to a <strong style="color:#fff">bell curve (normal)</strong> — great for modeling</li>
                <li>Most cars are priced between <strong style="color:#e8c547">$22,000–$28,000</strong></li>
                <li>Minimum price ~<strong style="color:#fff">$5,000</strong>, maximum ~<strong style="color:#fff">$42,000</strong></li>
                <li>No extreme outliers that require removal</li>
                <li>Dataset is fairly <strong style="color:#fff">balanced</strong> — not heavily skewed to one side</li>
            </ul>
            <br>
            <h4>🤖 Model Implication</h4>
            <p>A near-normal distribution allows the model to learn price patterns more easily without additional transformations.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image('images/hist.png', use_column_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Heatmap ──────────────────────────────────────────────
    st.markdown('<div class="card-title">🌡️ Heatmap — Correlation Between Numerical Features</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1], gap="large")
    with col1:
        st.image('images/heatmap.png', use_column_width=True)
    with col2:
        st.markdown("""
        <div class="desc-block">
            <h4>📌 What This Shows</h4>
            <p>The heatmap shows <strong style="color:#fff">correlation values</strong> between numerical columns. Values near <strong style="color:#e8c547">1.0</strong> = strong positive, near <strong style="color:#e8c547">-1.0</strong> = strong negative.</p>
            <br>
            <h4>💡 Key Insights</h4>
            <ul>
                <li><strong style="color:#fff">Year vs Price = 0.61</strong> — newer cars tend to be more expensive ✅</li>
                <li><strong style="color:#fff">Mileage vs Price = -0.56</strong> — high mileage = lower price ✅</li>
                <li><strong style="color:#fff">Engine Size vs Price = 0.38</strong> — bigger engine = slightly higher price ✅</li>
                <li>Year and Mileage are nearly <strong style="color:#fff">uncorrelated (0.016)</strong> — both provide unique information to the model</li>
            </ul>
            <br>
            <h4>🤖 Model Implication</h4>
            <p><strong style="color:#e8c547">Year</strong> and <strong style="color:#e8c547">Mileage</strong> are the two most important features. No high multicollinearity detected — all features are safe to use together.</p>
        </div>
        """, unsafe_allow_html=True)

# ── PAGE 4: DESCRIPTION ──────────────────────────────────────
elif menu == "📖 Description":
    st.markdown('<div class="hero-title">About This<br><span class="accent">Project</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Everything you need to know about this car price prediction model</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="desc-block">
            <h4>🎯 Project Goal</h4>
            <p>This app predicts the market price of a used car based on its specifications using a trained Machine Learning model. Built for learning purposes to demonstrate a full ML pipeline from data cleaning to deployment.</p>
        </div>
        <div class="desc-block">
            <h4>📦 Dataset</h4>
            <ul>
                <li>1,000 car records</li>
                <li>Features: Make, Year, Engine Size, Mileage, Fuel Type, Transmission</li>
                <li>Target: Price (USD)</li>
                <li>Source: Synthetic dataset for learning purposes</li>
            </ul>
        </div>
        <div class="desc-block">
            <h4>🤖 Why Random Forest?</h4>
            <ul>
                <li>Handles non-linear relationships well</li>
                <li>Robust to outliers</li>
                <li>Works with mixed feature types</li>
                <li>No scaling required for numerical features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="desc-block">
            <h4>🔧 ML Pipeline</h4>
            <ul>
                <li><strong style="color:#fff">Passthrough</strong> — Year, Engine Size, Mileage passed as-is</li>
                <li><strong style="color:#fff">OneHotEncoder</strong> — Make, Fuel Type, Transmission encoded to binary</li>
                <li><strong style="color:#fff">RandomizedSearchCV</strong> — 20 iterations, 5-fold cross validation</li>
                <li><strong style="color:#fff">Pickle</strong> — model serialized for deployment</li>
            </ul>
        </div>
        <div class="desc-block">
            <h4>📊 Final Performance</h4>
            <ul>
                <li>R² Test: <strong style="color:#e8c547">0.804</strong></li>
                <li>MAE: <strong style="color:#e8c547">$1,868</strong></li>
                <li>RMSE: <strong style="color:#e8c547">$2,315</strong></li>
                <li>MAPE: <strong style="color:#e8c547">7.98%</strong></li>
                <li>Train–Test Gap: <strong style="color:#e8c547">~10%</strong></li>
            </ul>
        </div>
        <div class="desc-block">
            <h4>🛠️ Tech Stack</h4>
            <ul>
                <li>Python, Pandas, Scikit-learn</li>
                <li>Streamlit (deployment)</li>
                <li>Pickle (model serialization)</li>
                <li>Matplotlib & Seaborn (EDA)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)