import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Cấu hình giao diện
st.set_page_config(page_title="VisionSense - Student Support System", layout="wide")

# --- HÀM LOAD DỮ LIỆU ---
@st.cache_data
def load_all_data():
    with open('student_records20.json', 'r') as f:
        records = json.load(f)
    with open('evaluation_results_20full_f.json', 'r') as f:
        evals = json.load(f)
    with open('llm_feedback_20_full.json', 'r') as f:
        reports = json.load(f)
    with open('questions_schema.json', 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return records, evals, reports, schema

try:
    records, evals, reports, schema = load_all_data()
    
    # --- SIDEBAR: CHỌN SINH VIÊN ---
    # st.sidebar.image("https://upload.wikimedia.org/wikipedia/vi/e/e5/Logo_Đại_học_Công_nghệ%2C_Đại_học_Quốc_gia_Hà_Nội.lang.png", width=100)
    st.sidebar.title("Hệ thống Dự đoán Sớm")
    student_ids = [r['student_id'] for r in records]
    selected_id = st.sidebar.selectbox("Chọn Mã sinh viên (Student ID):", student_ids)

    # Lấy dữ liệu cụ thể của sinh viên được chọn
    curr_record = next(item for item in records if item['student_id'] == selected_id)
    curr_eval = next(item for item in evals if item['student_id'] == selected_id)
    curr_report = next(item for item in reports if item['student_id'] == selected_id)

    # --- GIAO DIỆN CHÍNH ---
    st.title(f"Phân tích Học tập Cá nhân hóa - ID: {selected_id}")
    st.markdown(f"**Mốc thời gian dự báo:** {curr_record['marker']*100}% khóa học (Tuần {curr_record['cutoff_week']})")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        # 1. Gauge Chart dự đoán rủi ro
        prob = curr_record['y_prob']
        risk_color = "red" if curr_record['y_pred'] == 0 else "green"
        status = "CÓ RỦI RO (FAIL)" if curr_record['y_pred'] == 0 else "AN TOÀN (PASS)"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': f"Xác suất PASS: {status}"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # 2. Bảng điểm đánh giá LLM-as-a-judge
        st.subheader("Chất lượng Phản hồi (Judge Score)")
        acc_ana = curr_eval['analysis_eval']['accuracy']
        acc_feed = curr_eval['feedback_eval']['accuracy']
        
        st.metric("Analysis Accuracy", f"{acc_ana*100:.1f}%")
        st.metric("Feedback Accuracy", f"{acc_feed*100:.1f}%")
        
        with st.expander("Chi tiết các tiêu chí đạt (YES/NO)"):
            def create_dq_df(questions, scores):
                # Map câu hỏi từ JSON với điểm số tương ứng
                df = pd.DataFrame({
                    "Tiêu chí đánh giá (DQ)": questions,
                    "Kết quả": scores
                })
                df['Trạng thái'] = df['Kết quả'].apply(lambda x: "✅" if x == "YES" else "❌")
                return df[["Tiêu chí đánh giá (DQ)", "Trạng thái"]]

            # Giai đoạn Phân tích
            st.write("**Giai đoạn 1: Phân tích (Analysis)**")
            # Map trực tiếp từ schema['analysis']
            df_ana = create_dq_df(schema['analysis'], curr_eval['analysis_eval']['scores'])
            st.table(df_ana)

            st.divider()

            # Giai đoạn Phản hồi
            st.write("**Giai đoạn 2: Phản hồi (Feedback)**")
            # Map trực tiếp từ schema['feedback']
            df_feed = create_dq_df(schema['feedback'], curr_eval['feedback_eval']['scores'])
            st.table(df_feed)

    with col2:
        # 3. Hiển thị SHAP (Dạng bảng đơn giản cho nhanh)
        st.subheader("Tại sao mô hình dự đoán như vậy? (SHAP)")
        shap_df = pd.DataFrame({
            "Đặc trưng": curr_record['feature_names'],
            "Giá trị thực": curr_record['feature_values'],
            "Mức độ đóng góp (SHAP)": curr_record['shap_values']
        }).sort_values(by="Mức độ đóng góp (SHAP)", ascending=False)
        
        # Chỉ lấy top 5 ảnh hưởng nhất
        st.dataframe(shap_df.head(5), use_container_width=True)
        st.caption("Giá trị SHAP dương đẩy dự đoán về phía PASS, giá trị âm đẩy về phía FAIL.")

        # 4. Nội dung báo cáo và Phản hồi AI
        tab1, tab2 = st.tabs(["Báo cáo Phân tích (Analysis)", "Phản hồi gửi Sinh viên (Feedback)"])
        
        with tab1:
            st.markdown(curr_report['analysis'])
            
        with tab2:
            st.info(curr_report['feedback'])

except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy các file JSON. Hãy đảm bảo các file nằm cùng thư mục với app.py")
except Exception as e:
    st.exception(e)