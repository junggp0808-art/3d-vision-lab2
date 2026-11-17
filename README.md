# 3D Vision Lab (수정 버전)

## 이번 오류 수정
스크린샷의 `ModuleNotFoundError`는 앱이 `cv2`(OpenCV)를 불러오지 못해서 발생한 오류입니다.
이 프로젝트의 `requirements.txt`에 OpenCV 패키지를 명시했고, 앱에도 설치 누락 시 안내문을 표시하도록 했습니다.

## GitHub 업로드 방법
1. 기존 저장소 `3d-vision-lab`을 엽니다.
2. 기존의 잘못된 앱 파일(예: `app 2.py`) 대신 이 ZIP의 `app.py`를 업로드합니다.
3. 저장소 최상위(root)에 `requirements.txt`도 반드시 업로드합니다.
4. Streamlit 설정에서 Main file path가 정확히 `app.py`인지 확인합니다.
5. Streamlit Cloud에서 Manage app → Reboot를 누릅니다. 의존성 파일 변경 후에도 설치가 갱신되지 않으면 앱을 재배포합니다.

## 앱 실행
`streamlit run app.py`

## 입력
- 왼쪽 사진
- 오른쪽 사진
- 스테레오 보정 `.npz` 파일에 `K1`, `D1`, `K2`, `D2`, `R`, `T` 배열 필요

## 한계
이 코드는 보정된 스테레오 사진에서 ORB 특징점의 희소 3D 좌표를 추정하는 교육용 데모입니다. 임의의 사진 두 장만으로 인체 전체의 완전한 3D 메시를 만드는 기능은 아닙니다.
