import streamlit as st
from utils.inference import load_model, predict_image, predict_video
from PIL import Image
import tempfile
import os
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="🚘 Vehicle Detection System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .upload-box {
        border: 2px dashed #FF4B4B;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🚗 Hệ thống Nhận dạng Xe cộ")
    st.markdown("##### *Powered by YOLOv11 & YOLOv8*")

st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3097/3097161.png", width=100)
    st.header("⚙️ Cấu hình")
    
    model_choice = st.selectbox(
        "🤖 Chọn model:",
        ["YOLOv11 (best)", "YOLOv11 (last)", "YOLOv8 (best)", "YOLOv8 (last)"],
        help="Chọn model phát hiện xe phù hợp"
    )
    
    st.markdown("---")
    
    input_type = st.radio(
        "📁 Loại đầu vào:",
        ["Ảnh đơn", "Video", "Batch Test (Nhiều ảnh)"],
        help="Chọn loại dữ liệu đầu vào"
    )
    
    st.markdown("---")
    
    # Confidence threshold
    conf_threshold = st.slider(
        " Ngưỡng tin cậy:",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Chỉ hiển thị kết quả có độ tin cậy cao hơn ngưỡng này"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Thông tin")
    st.info("💡 **Tip**: Tải lên nhiều ảnh cùng lúc với chế độ Batch Test!")

# --- Load model ---
@st.cache_resource
def get_model(model_path):
    return load_model(model_path)

if model_choice == "YOLOv11 (best)":
    model_path = r"D:\nhan_dang_mau\runs\train\yolo11m_exp2\weights\best.pt"
    with st.sidebar:
        with st.spinner("Đang tải model..."):
            model = get_model(model_path)
        if model:
            st.success("✅ YOLOv11 (best) sẵn sàng!")
elif model_choice == "YOLOv11 (last)":
    model_path = r"D:\nhan_dang_mau\runs\train\yolo11m_exp2\weights\last.pt"
    with st.sidebar:
        with st.spinner("Đang tải model..."):
            model = get_model(model_path)
        if model:
            st.success("✅ YOLOv11 (last) sẵn sàng!")
elif model_choice == "YOLOv8 (best)":
    model_path = r"D:\nhan_dang_mau\runs\train\yolov8n_exp\weights\best.pt"
    with st.sidebar:
        with st.spinner("Đang tải model..."):
            model = get_model(model_path)
        if model:
            st.success("✅ YOLOv11 (last) sẵn sàng!")
elif model_choice == "YOLOv8 (last)":
    model_path = r"D:\nhan_dang_mau\runs\train\yolov8n_exp\weights\last.pt"
    with st.sidebar:
        with st.spinner("Đang tải model..."):
            model = get_model(model_path)
        if model:
            st.success("✅ YOLOv11 (last) sẵn sàng!")
else:
    model = None
    st.sidebar.warning("⚠️ Model chưa khả dụng")

# --- Main interface ---
if model is None:
    st.error("❌ **Model không khả dụng**")
    st.info("💡 Vui lòng thêm file `.pt` vào thư mục `models/` sau khi huấn luyện.")
    
else:
    # ========== ẢNH ĐƠN ==========
    if input_type == "Ảnh đơn":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Tải ảnh lên")
            uploaded_image = st.file_uploader(
                "Chọn ảnh xe cộ",
                type=["jpg", "jpeg", "png"],
                help="Hỗ trợ: JPG, JPEG, PNG"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="🖼️ Ảnh gốc", use_container_width=True)
                
                if st.button("🔍 Phát hiện xe"):
                    with st.spinner("Đang phân tích..."):
                        result_img, preds = predict_image(model, image, conf_threshold)
                        
                        # Lưu kết quả vào session state
                        st.session_state['result_img'] = result_img
                        st.session_state['preds'] = preds
        
        with col2:
            if 'result_img' in st.session_state:
                st.markdown("### ✨ Kết quả nhận dạng")
                st.image(st.session_state['result_img'], caption="🎯 Kết quả", use_container_width=True)
                
                # Hiển thị thống kê
                preds = st.session_state['preds']
                num_vehicles = len(preds)
                
                st.markdown(f"""
                    <div class="metric-card">
                        <h2>{num_vehicles}</h2>
                        <p>Phương tiện phát hiện</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### 📋 Chi tiết phát hiện")
                
                if num_vehicles > 0:
                    df = pd.DataFrame(preds, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                    df.index = df.index + 1
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("⚠️ Không phát hiện phương tiện nào")
    
    # ========== VIDEO ==========
    elif input_type == "Video":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Tải video lên")
            uploaded_video = st.file_uploader(
                "Chọn video xe cộ",
                type=["mp4", "mov", "avi", "mkv"],
                help="Hỗ trợ: MP4, MOV, AVI, MKV"
            )
            
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                st.video(tfile.name)
                
                if st.button("▶️ Phân tích video"):
                    with st.spinner("⏳ Đang xử lý video... (có thể mất vài phút)"):
                        output_path = predict_video(model, tfile.name, conf_threshold)
                        st.session_state['output_video'] = output_path
        
        with col2:
            if 'output_video' in st.session_state:
                st.markdown("### ✨ Video đã xử lý")
                st.video(st.session_state['output_video'])
                
                st.markdown("""
                    <div class="success-box">
                        <h4>✅ Hoàn tất phân tích video!</h4>
                        <p>Video đã được xử lý và phát hiện xe thành công.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Nút tải xuống
                with open(st.session_state['output_video'], 'rb') as f:
                    st.download_button(
                        label="💾 Tải video xuống",
                        data=f,
                        file_name=f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
    
    # ========== BATCH TEST ==========
    elif input_type == "Batch Test (Nhiều ảnh)":
        st.markdown("### 📊 Kiểm tra hàng loạt")
        st.info("💡 Tải lên nhiều ảnh để phân tích và so sánh kết quả")
        
        uploaded_files = st.file_uploader(
            "Chọn nhiều ảnh để test",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Có thể chọn nhiều file cùng lúc"
        )
        
        if uploaded_files:
            st.markdown(f"**📁 Đã tải lên: {len(uploaded_files)} ảnh**")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🚀 Bắt đầu phân tích hàng loạt", type="primary"):
                    st.session_state['batch_processing'] = True
            
            with col2:
                show_details = st.checkbox("📋 Hiển thị chi tiết", value=True)
            
            if st.session_state.get('batch_processing', False):
                # Tạo progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                # Xử lý từng ảnh
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Đang xử lý: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                    
                    image = Image.open(uploaded_file)
                    result_img, preds = predict_image(model, image, conf_threshold)
                    
                    results.append({
                        'file_name': uploaded_file.name,
                        'image': image,
                        'result_img': result_img,
                        'num_vehicles': len(preds),
                        'preds': preds
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                # Hiển thị tổng kết
                st.success(f"✅ **Hoàn tất!** Đã phân tích {len(uploaded_files)} ảnh")
                
                # Thống kê tổng quan
                col1, col2, col3 = st.columns(3)
                
                total_vehicles = sum([r['num_vehicles'] for r in results])
                avg_vehicles = total_vehicles / len(results) if results else 0
                max_vehicles = max([r['num_vehicles'] for r in results]) if results else 0
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{total_vehicles}</h3>
                            <p>Tổng phương tiện</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_vehicles:.1f}</h3>
                            <p>Trung bình/ảnh</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{max_vehicles}</h3>
                            <p>Nhiều nhất/ảnh</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Hiển thị kết quả từng ảnh
                st.markdown("### 📸 Kết quả chi tiết")
                
                for idx, result in enumerate(results):
                    with st.expander(f"🖼️ {result['file_name']} - Phát hiện: {result['num_vehicles']} xe", expanded=(idx==0)):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(result['image'], caption="Ảnh gốc", use_container_width=True)
                        
                        with col2:
                            st.image(result['result_img'], caption="Kết quả", use_container_width=True)
                        
                        if show_details and result['num_vehicles'] > 0:
                            st.markdown("**📋 Chi tiết:**")
                            df = pd.DataFrame(result['preds'], columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
                            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                            df.index = df.index + 1
                            st.dataframe(df, use_container_width=True)
                
                # Nút reset
                if st.button("🔄 Làm mới"):
                    st.session_state['batch_processing'] = False
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚗 Vehicle Detection System | Developed with ❤️ using Streamlit & YOLOv11</p>
    </div>
""", unsafe_allow_html=True)