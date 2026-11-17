import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="3D Vision Lab", page_icon="🔭", layout="wide")
st.title("🔭 3D Vision Lab")
st.subheader("두 시점의 사진으로 3차원 특징점 복원하기")

try:
    import cv2
except ModuleNotFoundError:
    st.error("필수 라이브러리 OpenCV(cv2)가 설치되지 않았습니다.")
    st.markdown("""
    **해결 방법**
    1. GitHub 저장소 최상위 폴더에 `requirements.txt`가 있는지 확인하세요.
    2. 파일 안에 `opencv-python-headless==4.11.0.86`가 있는지 확인하세요.
    3. Streamlit의 **Manage app → Reboot**를 누르거나 새 배포를 실행하세요.
    """)
    st.stop()

st.info("교육용 스테레오 비전 데모입니다. 보정된 좌우 카메라 사진으로 희소 3D 특징점만 복원하며, 완전한 인체 표면 메시를 생성하지는 않습니다.")

with st.sidebar:
    st.header("입력 파일")
    calib_file = st.file_uploader("스테레오 카메라 보정 파일 (.npz)", type=["npz"])
    left_file = st.file_uploader("왼쪽 카메라 사진", type=["jpg","jpeg","png"])
    right_file = st.file_uploader("오른쪽 카메라 사진", type=["jpg","jpeg","png"])
    max_features = st.slider("최대 특징점 수", 500, 6000, 3000, 500)
    ratio = st.slider("매칭 엄격도", 0.5, 0.9, 0.7, 0.05)
    max_gap = st.number_input("두 시선 사이 허용 거리 (보정 단위)", 0.01, 10.0, 1.0, 0.05)
    run = st.button("3차원 복원 시작", type="primary", use_container_width=True)

st.markdown("""
### 촬영 및 입력 조건
- 좌우 사진은 같은 장면을 서로 다른 위치에서 촬영해야 합니다.
- 가능하면 두 카메라를 고정하고 같은 순간에 촬영하세요.
- 카메라 보정에 사용한 카메라·해상도·초점 설정을 유지해야 합니다.
- `calibration.npz`에는 `K1`, `D1`, `K2`, `D2`, `R`, `T` 배열이 있어야 합니다.
""")

def decode_image(upload):
    raw = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지 파일을 읽을 수 없습니다.")
    return image

def reconstruct(calib, left, right, nfeatures, ratio_test, max_ray_gap):
    keys = ["K1", "D1", "K2", "D2", "R", "T"]
    missing = [k for k in keys if k not in calib]
    if missing:
        raise ValueError("보정 파일에 다음 항목이 없습니다: " + ", ".join(missing))
    K1,D1,K2,D2,R,T = [calib[k] for k in keys]
    T = np.asarray(T, dtype=float).reshape(3, 1)
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("좌우 사진의 가로·세로 크기가 같아야 합니다.")

    g1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=int(nfeatures))
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None:
        raise ValueError("특징점을 찾지 못했습니다. 질감이나 모서리가 있는 장면을 사용하세요.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair)==2 for m,n in [pair] if m.distance < ratio_test*n.distance]
    if len(good) < 8:
        raise ValueError(f"대응 특징점이 부족합니다 ({len(good)}개). 사진을 다시 촬영해 주세요.")

    p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    q1 = cv2.undistortPoints(p1, K1, D1).reshape(-1,2)
    q2 = cv2.undistortPoints(p2, K2, D2).reshape(-1,2)

    # 정규화 좌표에 대한 기본행렬로 잘못된 대응점을 제거
    F, mask = cv2.findFundamentalMat(q1, q2, cv2.FM_RANSAC, 0.001, 0.99)
    if F is None or mask is None:
        raise ValueError("두 사진의 기하학적 대응 관계를 검증하지 못했습니다.")
    inliers = mask.ravel().astype(bool)
    q1, q2 = q1[inliers], q2[inliers]

    # 좌표계: 카메라 1이 원점, 카메라 2의 좌표 변환은 X2 = R X1 + T
    C1 = np.zeros(3)
    C2 = (-R.T @ T).reshape(3)
    points, gaps, angles = [], [], []
    for a,b in zip(q1,q2):
        d1 = np.array([a[0], a[1], 1.0])
        d2_cam2 = np.array([b[0], b[1], 1.0])
        u = d1 / np.linalg.norm(d1)
        v = R.T @ d2_cam2
        v = v / np.linalg.norm(v)
        w = C1 - C2
        c = float(np.dot(u,v)); d = float(np.dot(w,u)); e = float(np.dot(w,v))
        den = 1.0 - c*c
        if den < 1e-9:
            continue
        s = (c*e-d)/den
        t = (e-c*d)/den
        if s <= 0 or t <= 0:
            continue
        P1 = C1 + s*u
        P2 = C2 + t*v
        gap = float(np.linalg.norm(P1-P2))
        if gap <= max_ray_gap:
            points.append((P1+P2)/2)
            gaps.append(gap)
            angles.append(float(np.degrees(np.arccos(np.clip(abs(c),0,1)))))
    if not points:
        raise ValueError("복원된 점이 없습니다. 보정 파일, 촬영 조건, 허용 거리를 확인하세요.")
    df = pd.DataFrame(np.asarray(points), columns=["X","Y","Z"])
    df["ray_gap"] = gaps
    df["ray_angle_deg"] = angles
    return df, len(good), int(inliers.sum())

if run:
    if not all([calib_file, left_file, right_file]):
        st.warning("보정 파일과 좌우 사진을 모두 업로드해 주세요.")
    else:
        try:
            with np.load(calib_file) as loaded:
                calib = {k: loaded[k] for k in loaded.files}
            left = decode_image(left_file)
            right = decode_image(right_file)
            with st.spinner("특징점 매칭과 3차원 좌표 계산 중입니다..."):
                df, matched, inlier_count = reconstruct(calib,left,right,max_features,ratio,max_gap)
            st.success("복원이 완료되었습니다.")
            a,b,c = st.columns(3)
            a.metric("초기 매칭", matched)
            b.metric("기하 검증 통과", inlier_count)
            c.metric("최종 3D 점", len(df))
            fig = go.Figure(data=[go.Scatter3d(
                x=df.X, y=df.Y, z=df.Z, mode="markers",
                marker=dict(size=3, color=df.Z, colorscale="Viridis", opacity=0.85),
                text=[f"X={r.X:.3f}, Y={r.Y:.3f}, Z={r.Z:.3f}" for r in df.itertuples()],
                hoverinfo="text"
            )])
            fig.update_layout(
                title="복원된 3차원 특징점",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                height=650, margin=dict(l=0,r=0,b=0,t=45)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("좌표 데이터")
            st.dataframe(df, use_container_width=True)
            st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                               "3d_coordinates.csv", "text/csv")
            st.caption("좌표 단위는 카메라 보정 파일을 만들 때 설정한 기준 길이 단위를 따릅니다.")
        except Exception as exc:
            st.error(f"복원 중 문제가 발생했습니다: {exc}")

st.divider()
st.caption("개인정보 보호를 위해 본인 또는 촬영 동의를 받은 사진만 사용하세요.")
